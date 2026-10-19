"""Run WiLoR on rectified pinhole videos and return full-frame 21-point observations.

WiLoR's wrist-relative 3D is stored only as audit data. This module does not
read HumanPlus directories, rectify fish-eye video, triangulate, or render.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts import KEYPOINT_COUNT, PIXEL_SPACE_RECTIFIED_PINHOLE, CameraKeypoints21, RectifiedStereoForWiLoR


@dataclass(frozen=True)
class WiLoRExtractionConfig:
    """WiLoR runtime settings. The backend stays WiLoR-mini; nothing is substituted.

    Args:
        device: Torch device string, normally ``"cuda"``. ``None`` selects CUDA when
            present and otherwise CPU, and the chosen device is recorded by the caller.
        minimum_detection_confidence: Inclusive threshold forwarded to WiLoR-mini.
        maximum_hands_per_frame: How many hand slots to keep. Extra detections are dropped
            after sorting by the detector's order, not replaced with invented points.
    """

    device: str | None = "cuda"
    minimum_detection_confidence: float = 0.30
    maximum_hands_per_frame: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_detection_confidence <= 1.0:
            raise ValueError("minimum_detection_confidence must be in [0, 1].")
        if self.maximum_hands_per_frame != 2:
            raise ValueError("This pipeline keeps exactly two hand slots per frame.")


def extract_wilor(
    capture: RectifiedStereoForWiLoR, config: WiLoRExtractionConfig
) -> tuple[CameraKeypoints21, CameraKeypoints21]:
    """Detect 21 keypoints on the left and right rectified videos.

    Args:
        capture: Output of HumanPlus preparation. Its pixel space must be rectified pinhole.
        config: Device and detection threshold for the official WiLoR-mini pipeline.
    Returns:
        Left and right ``CameraKeypoints21`` aligned to ``capture`` timestamps.
    Raises:
        ValueError: If the capture is not rectified pinhole.
        RuntimeError: If WiLoR-mini is missing or a frame returns a malformed 21-point array.
    """

    if capture.pixel_space != PIXEL_SPACE_RECTIFIED_PINHOLE:
        raise ValueError("extract_wilor refuses any pixel space other than RECTIFIED_PINHOLE.")
    backend = _WiLoRMiniBackend(config.device, config.minimum_detection_confidence)
    left = _extract_one_video(capture.left_pinhole.camera_id, capture.left_rectified_video, capture, backend, config)
    right = _extract_one_video(capture.right_pinhole.camera_id, capture.right_rectified_video, capture, backend, config)
    return left, right


class _WiLoRMiniBackend:
    """Thin adapter over WiLoR-mini. Full-frame 2D pixels are the geometric output."""

    def __init__(self, device: str | None, minimum_detection_confidence: float) -> None:
        try:
            import torch
            from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
                WiLorHandPose3dEstimationPipeline,
            )
        except ImportError as error:
            raise RuntimeError(
                "WiLoR-mini is not installed. Install the official wilor_mini package and weights first."
            ) from error
        if device is None:
            selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            selected = torch.device(device)
            if selected.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("WiLoRExtractionConfig.device is 'cuda', but CUDA is not available.")
        dtype = torch.float16 if selected.type != "cpu" else torch.float32
        self.minimum_detection_confidence = minimum_detection_confidence
        self.device_name = str(selected)
        self._pipeline = WiLorHandPose3dEstimationPipeline(device=selected, dtype=dtype, verbose=False)

    def infer(self, image_rgb: np.ndarray) -> list[dict[str, np.ndarray | str | float]]:
        """Return up to the detector's hands. 2D points are already full-frame pixels."""

        predictions: list[dict[str, np.ndarray | str | float]] = []
        for output in self._pipeline.predict(image_rgb, hand_conf=self.minimum_detection_confidence):
            model_prediction = output.get("wilor_preds")
            if model_prediction is None:
                continue
            points_px = np.asarray(model_prediction["pred_keypoints_2d"][0], dtype=np.float64)
            if points_px.shape != (KEYPOINT_COUNT, 2):
                raise RuntimeError(f"WiLoR returned keypoints with shape {points_px.shape}; expected (21, 2).")
            relative_3d = np.full((KEYPOINT_COUNT, 3), np.nan, dtype=np.float64)
            if "pred_keypoints_3d" in model_prediction:
                candidate = np.asarray(model_prediction["pred_keypoints_3d"][0], dtype=np.float64)
                if candidate.shape == (KEYPOINT_COUNT, 3):
                    relative_3d = candidate
            is_right = bool(float(output.get("is_right", 0.0)))
            predictions.append(
                {
                    "points_px": points_px,
                    "model_relative_3d_m": relative_3d,
                    "handedness": "right" if is_right else "left",
                }
            )
        return predictions


def _extract_one_video(camera_id, video_path, capture, backend: _WiLoRMiniBackend, config: WiLoRExtractionConfig):
    cv2 = _require_cv2()
    frame_count = len(capture.frame_indices)
    points_px = np.full((frame_count, 2, KEYPOINT_COUNT, 2), np.nan, dtype=np.float64)
    confidence = np.zeros((frame_count, 2, KEYPOINT_COUNT), dtype=np.float64)
    detection_confidence = np.zeros((frame_count, 2), dtype=np.float64)
    handedness = np.full((frame_count, 2), "", dtype="U16")
    handedness_confidence = np.zeros((frame_count, 2), dtype=np.float64)
    hand_present = np.zeros((frame_count, 2), dtype=bool)
    relative_3d = np.full((frame_count, 2, KEYPOINT_COUNT, 3), np.nan, dtype=np.float64)
    status = np.full(frame_count, "no_hand", dtype="U32")
    status_reason = np.full(frame_count, "", dtype="U128")

    decoder = cv2.VideoCapture(str(video_path))
    if not decoder.isOpened():
        raise FileNotFoundError(f"Cannot open rectified video: {video_path}")
    try:
        for frame_index in range(frame_count):
            read_ok, frame_bgr = decoder.read()
            if not read_ok or frame_bgr is None:
                status[frame_index] = "frame_unavailable"
                status_reason[frame_index] = "rectified_video_ended"
                continue
            image_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            try:
                predictions = backend.infer(image_rgb)
            except (RuntimeError, ValueError) as error:
                status[frame_index] = "backend_error"
                status_reason[frame_index] = str(error)[:128]
                continue
            kept = predictions[: config.maximum_hands_per_frame]
            if not kept:
                status[frame_index] = "no_hand"
                continue
            for slot, prediction in enumerate(kept):
                points_px[frame_index, slot] = prediction["points_px"]
                confidence[frame_index, slot] = 1.0
                detection_confidence[frame_index, slot] = 1.0
                handedness[frame_index, slot] = str(prediction["handedness"])
                handedness_confidence[frame_index, slot] = 1.0
                hand_present[frame_index, slot] = True
                relative_3d[frame_index, slot] = prediction["model_relative_3d_m"]
            status[frame_index] = "success"
            if len(predictions) > config.maximum_hands_per_frame:
                status_reason[frame_index] = "extra_hands_dropped"
    finally:
        decoder.release()

    return CameraKeypoints21(
        camera_id=camera_id,
        pixel_space=PIXEL_SPACE_RECTIFIED_PINHOLE,
        frame_indices=np.asarray(capture.frame_indices, dtype=np.int64),
        timestamps_ns=np.asarray(capture.timestamps_ns, dtype=np.int64),
        points_px=points_px,
        keypoint_confidence=confidence,
        detection_confidence=detection_confidence,
        handedness=handedness,
        handedness_confidence=handedness_confidence,
        hand_present=hand_present,
        status=status,
        status_reason=status_reason,
        model_relative_3d_m=relative_3d,
        backend_name=f"wilor_mini:{backend.device_name}",
    )


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to read rectified video.") from error
    return cv2
