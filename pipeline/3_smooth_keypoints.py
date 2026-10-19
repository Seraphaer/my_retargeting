"""Smooth WiLoR 2D pixels and wrist-relative 3D before triangulation.

This is the ssh_run_all ``HandKeypointSmoother``: one One-Euro filter per
anatomical hand, on normalized image coordinates and on the wrist-relative
3D points. A missing hand resets that hand's filters. The smoothed relative
3D stays audit data and does not enter triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts import KEYPOINT_COUNT, CameraKeypoints21


HAND_SIDES = ("left", "right")


@dataclass(frozen=True)
class SmoothConfig:
    """SSH detection-stage One-Euro parameters.

    Args:
        minimum_cutoff_hz: Stationary cutoff. SSH uses ``1.0``.
        beta: Speed gain. SSH uses ``0.3``.
        derivative_cutoff_hz: Cutoff of the speed estimate. SSH uses ``1.0``.
    """

    minimum_cutoff_hz: float = 1.0
    beta: float = 0.3
    derivative_cutoff_hz: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_cutoff_hz <= 0.0 or self.derivative_cutoff_hz <= 0.0 or self.beta < 0.0:
            raise ValueError("One-Euro cutoffs must be positive and beta must be non-negative.")


def smooth_camera_keypoints(
    left: CameraKeypoints21,
    right: CameraKeypoints21,
    left_video: Path,
    right_video: Path,
    config: SmoothConfig,
) -> tuple[CameraKeypoints21, CameraKeypoints21]:
    """Return new observations with SSH One-Euro applied independently per camera.

    Args:
        left: Raw left-camera WiLoR observations. Slot order may follow the detector.
        right: Raw right-camera WiLoR observations.
        left_video: Rectified video whose size and frame rate define left-camera time.
        right_video: Rectified video whose size and frame rate define right-camera time.
        config: One-Euro cutoffs. Time is ``frame_index / fps``, as in SSH.
    Returns:
        Left and right observations rebinned to anatomical slots ``0=left``,
        ``1=right``. Absent hands stay absent. Raw inputs are not modified.
    """

    return (
        _smooth_one_camera(left, *_video_timing(left_video), config),
        _smooth_one_camera(right, *_video_timing(right_video), config),
    )


def rectified_video_fps(video_path: Path) -> float:
    """Return the video frame rate SSH uses, falling back to 30 when unset."""

    return _video_timing(video_path)[0]


class _OneEuroFilter:
    """Casiez et al. 2012, copied from ssh_run_all ``OneEuroFilter``."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, time_s: float, value: np.ndarray) -> np.ndarray:
        if self._t_prev is None or self._x_prev is None or self._dx_prev is None:
            self._x_prev = value.copy()
            self._dx_prev = np.zeros_like(value)
            self._t_prev = time_s
            return value.copy()
        dt = max(time_s - self._t_prev, 1e-6)
        dx = (value - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * value + (1 - alpha) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = time_s
        return x_hat.copy()


def _smooth_one_camera(
    keypoints: CameraKeypoints21,
    fps: float,
    image_size_px: tuple[int, int],
    config: SmoothConfig,
) -> CameraKeypoints21:
    frame_count = len(keypoints.frame_indices)
    width_px, height_px = image_size_px
    scale = np.array([width_px, height_px], dtype=np.float64)
    points_px = np.full((frame_count, 2, KEYPOINT_COUNT, 2), np.nan, dtype=np.float64)
    confidence = np.zeros((frame_count, 2, KEYPOINT_COUNT), dtype=np.float64)
    detection_confidence = np.zeros((frame_count, 2), dtype=np.float64)
    handedness = np.full((frame_count, 2), "", dtype="U16")
    handedness_confidence = np.zeros((frame_count, 2), dtype=np.float64)
    hand_present = np.zeros((frame_count, 2), dtype=bool)
    relative_3d = np.full((frame_count, 2, KEYPOINT_COUNT, 3), np.nan, dtype=np.float64)
    status = np.array(keypoints.status, copy=True)
    status_reason = np.array(keypoints.status_reason, copy=True)
    image_filters = {
        side: _OneEuroFilter(config.minimum_cutoff_hz, config.beta, config.derivative_cutoff_hz) for side in HAND_SIDES
    }
    world_filters = {
        side: _OneEuroFilter(config.minimum_cutoff_hz, config.beta, config.derivative_cutoff_hz) for side in HAND_SIDES
    }

    for frame_index in range(frame_count):
        time_s = frame_index / fps
        present_any = False
        for slot, side in enumerate(HAND_SIDES):
            source_slot = _first_slot_for_side(keypoints, frame_index, side)
            if source_slot is None:
                image_filters[side].reset()
                world_filters[side].reset()
                continue
            pixels = np.asarray(keypoints.points_px[frame_index, source_slot], dtype=np.float64)
            image_norm = np.concatenate([pixels / scale, np.zeros((KEYPOINT_COUNT, 1))], axis=1)
            smoothed_image = image_filters[side](time_s, image_norm)
            points_px[frame_index, slot] = smoothed_image[:, :2] * scale
            world = np.asarray(keypoints.model_relative_3d_m[frame_index, source_slot], dtype=np.float64)
            if np.isfinite(world).all():
                relative_3d[frame_index, slot] = world_filters[side](time_s, world)
            else:
                world_filters[side].reset()
                relative_3d[frame_index, slot] = world
            confidence[frame_index, slot] = keypoints.keypoint_confidence[frame_index, source_slot]
            detection_confidence[frame_index, slot] = keypoints.detection_confidence[frame_index, source_slot]
            handedness[frame_index, slot] = side
            handedness_confidence[frame_index, slot] = keypoints.handedness_confidence[frame_index, source_slot]
            hand_present[frame_index, slot] = True
            present_any = True
        if str(status[frame_index]) in {"frame_unavailable", "backend_error"}:
            continue
        status[frame_index] = "success" if present_any else "no_hand"
        status_reason[frame_index] = ""
    return CameraKeypoints21(
        camera_id=keypoints.camera_id,
        pixel_space=keypoints.pixel_space,
        frame_indices=np.asarray(keypoints.frame_indices, dtype=np.int64),
        timestamps_ns=np.asarray(keypoints.timestamps_ns, dtype=np.int64),
        points_px=points_px,
        keypoint_confidence=confidence,
        detection_confidence=detection_confidence,
        handedness=handedness,
        handedness_confidence=handedness_confidence,
        hand_present=hand_present,
        status=status,
        status_reason=status_reason,
        model_relative_3d_m=relative_3d,
        backend_name=f"{keypoints.backend_name}:one_euro",
    )


def _first_slot_for_side(keypoints: CameraKeypoints21, frame_index: int, side: str) -> int | None:
    """Keep the first detection of one anatomical hand, as SSH does."""

    for slot in range(2):
        if bool(keypoints.hand_present[frame_index, slot]) and str(keypoints.handedness[frame_index, slot]) == side:
            return slot
    return None


def _video_timing(video_path: Path) -> tuple[float, tuple[int, int]]:
    """Read FPS and pixel size the same way SSH opens a detection video."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open rectified video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width_px = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_px = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0.0:
        fps = 30.0
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Rectified video has no positive size: {video_path}")
    return fps, (width_px, height_px)
