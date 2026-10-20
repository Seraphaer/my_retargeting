"""Frozen data passed between HumanPlus preparation, WiLoR extraction, and post-processing.

This module validates stage boundaries and serializes them. It does not read
HumanPlus files, run WiLoR, triangulate, filter, or draw video.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


KEYPOINT_COUNT = 21
PIXEL_SPACE_RECTIFIED_PINHOLE = "RECTIFIED_PINHOLE"
COORDINATE_FRAME_LEFT_RECTIFIED = "left_rectified_camera"
CAPTURE_FORMAT = "rectified_stereo_for_wilor"
CAPTURE_VERSION = 1

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class CameraModel:
    """One camera's pixel model and its pose in the HumanPlus rig.

    Args:
        camera_id: Dataset camera name, such as ``fisheye_left``.
        camera_model: ``opencv_fisheye`` for the raw video, or ``rectified_pinhole``
            after preparation.
        image_size_px: ``(width_px, height_px)`` of the image this model describes.
        intrinsics: Row-major 3x3 camera matrix, in pixels.
        distortion: OpenCV fish-eye coefficients. Empty for a rectified pinhole.
        T_rig_from_camera: 4x4 transform from this camera's coordinates into the
            original rig frame. Translation is meters.
    """

    camera_id: str
    camera_model: str
    image_size_px: tuple[int, int]
    intrinsics: Matrix3
    distortion: tuple[float, ...]
    T_rig_from_camera: Matrix4

    def __post_init__(self) -> None:
        if self.image_size_px[0] <= 0 or self.image_size_px[1] <= 0:
            raise ValueError("image_size_px must contain positive width and height.")
        _require_shape(np.asarray(self.intrinsics, dtype=float), (3, 3), "intrinsics")
        _require_shape(np.asarray(self.T_rig_from_camera, dtype=float), (4, 4), "T_rig_from_camera")
        if self.camera_model == "rectified_pinhole" and self.distortion:
            raise ValueError("A rectified pinhole camera cannot carry fish-eye distortion.")
        if self.camera_model != "rectified_pinhole" and len(self.distortion) != 4:
            raise ValueError("A raw HumanPlus fish-eye camera requires four distortion coefficients.")


@dataclass(frozen=True)
class RectifiedStereoForWiLoR:
    """WiLoR-ready stereo videos plus the calibrations later stages must not recompute.

    Rectified videos are the only images WiLoR may read. Raw videos and raw
    fish-eye models stay attached so the acceptance renderer can project back
    without opening the HumanPlus annotation again.
    """

    session_id: str
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    is_synchronized: np.ndarray
    sync_reason: np.ndarray
    left_rectified_video: Path
    right_rectified_video: Path
    left_pinhole: CameraModel
    right_pinhole: CameraModel
    left_raw_video: Path
    right_raw_video: Path
    left_raw_fisheye: CameraModel
    right_raw_fisheye: CameraModel
    pixel_space: str = PIXEL_SPACE_RECTIFIED_PINHOLE

    def __post_init__(self) -> None:
        if self.pixel_space != PIXEL_SPACE_RECTIFIED_PINHOLE:
            raise ValueError("WiLoR input must already be RECTIFIED_PINHOLE.")
        count = _require_timeline(self.frame_indices, self.timestamps_ns, "rectified capture")
        if self.is_synchronized.shape != (count,) or self.sync_reason.shape != (count,):
            raise ValueError("Synchronization flags must align with the capture timeline.")
        if self.left_pinhole.camera_model != "rectified_pinhole" or self.right_pinhole.camera_model != "rectified_pinhole":
            raise ValueError("WiLoR cameras must use the rectified pinhole model.")


@dataclass(frozen=True)
class CameraKeypoints21:
    """WiLoR observations for one rectified camera. Model 3D is audit-only.

    ``points_px`` is ``(frame, hand_slot, 21, 2)`` in the rectified image.
    Empty slots have ``hand_present`` false. ``model_relative_3d_m`` is the
    wrist-relative WiLoR estimate in meters and must never enter triangulation.
    """

    camera_id: str
    pixel_space: str
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    points_px: np.ndarray
    keypoint_confidence: np.ndarray
    detection_confidence: np.ndarray
    handedness: np.ndarray
    handedness_confidence: np.ndarray
    hand_present: np.ndarray
    status: np.ndarray
    status_reason: np.ndarray
    model_relative_3d_m: np.ndarray
    backend_name: str

    def __post_init__(self) -> None:
        if self.pixel_space != PIXEL_SPACE_RECTIFIED_PINHOLE:
            raise ValueError("WiLoR keypoints must stay in RECTIFIED_PINHOLE pixels.")
        count = _require_timeline(self.frame_indices, self.timestamps_ns, self.camera_id)
        expected = {
            "points_px": (count, 2, KEYPOINT_COUNT, 2),
            "keypoint_confidence": (count, 2, KEYPOINT_COUNT),
            "detection_confidence": (count, 2),
            "handedness": (count, 2),
            "handedness_confidence": (count, 2),
            "hand_present": (count, 2),
            "status": (count,),
            "status_reason": (count,),
            "model_relative_3d_m": (count, 2, KEYPOINT_COUNT, 3),
        }
        for name, shape in expected.items():
            _require_shape(getattr(self, name), shape, f"{self.camera_id}.{name}")


@dataclass(frozen=True)
class HandTrack3D:
    """One anatomical hand after stereo triangulation, before whole-frame fill.

    Coordinates are meters in the left rectified camera. Invalid points are
    zeros, matching the SSH triangulation array, and ``point_valid_mask`` says
    which of those zeros are real measurements.
    """

    side: str
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    points_m: np.ndarray
    point_valid_mask: np.ndarray
    reprojection_error_px: np.ndarray
    frame_status: np.ndarray
    failure_reason: np.ndarray
    coordinate_frame: str = COORDINATE_FRAME_LEFT_RECTIFIED

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("Hand side must be 'left' or 'right'.")
        if self.coordinate_frame != COORDINATE_FRAME_LEFT_RECTIFIED:
            raise ValueError("Triangulated hands must stay in the left rectified camera.")
        count = _require_timeline(self.frame_indices, self.timestamps_ns, self.side)
        _require_shape(self.points_m, (count, KEYPOINT_COUNT, 3), f"{self.side}.points_m")
        _require_shape(self.point_valid_mask, (count, KEYPOINT_COUNT), f"{self.side}.point_valid_mask")
        _require_shape(self.reprojection_error_px, (count, KEYPOINT_COUNT), f"{self.side}.reprojection_error_px")
        _require_shape(self.frame_status, (count,), f"{self.side}.frame_status")
        _require_shape(self.failure_reason, (count,), f"{self.side}.failure_reason")


@dataclass(frozen=True)
class RepairedHandTrack:
    """One hand after SSH whole-frame gating and gap-aware fill.

    ``raw`` stays the triangulated measurement. ``invalid_mask`` is the
    frame-level mask consumed by the fill. ``points_m`` is the filled pose,
    so an invalid frame can still contain a held or interpolated hand.
    """

    raw: HandTrack3D
    points_m: np.ndarray
    invalid_mask: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.raw.frame_indices)
        _require_shape(self.points_m, (count, KEYPOINT_COUNT, 3), "filled.points_m")
        _require_shape(self.invalid_mask, (count,), "filled.invalid_mask")


@dataclass(frozen=True)
class RepairedHands:
    """Both hands after module 5, sharing the triangulation timeline."""

    left: RepairedHandTrack
    right: RepairedHandTrack

    def __post_init__(self) -> None:
        if self.left.raw.side != "left" or self.right.raw.side != "right":
            raise ValueError("RepairedHands requires one left track and one right track.")
        if not np.array_equal(self.left.raw.frame_indices, self.right.raw.frame_indices):
            raise ValueError("Repaired hands must share frame indices.")
        if not np.array_equal(self.left.raw.timestamps_ns, self.right.raw.timestamps_ns):
            raise ValueError("Repaired hands must share timestamps.")


@dataclass(frozen=True)
class GripperPoseTrack:
    """One hand's gripper pose in the left rectified camera.

    ``opening_distance_m`` is the thumb-index length in meters. ``opening_width``
    stays NaN until module 7 normalizes it into ``[0, 1]``.
    """

    side: str
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    opening_distance_m: np.ndarray
    opening_width: np.ndarray
    invalid_mask: np.ndarray
    jaw_degenerate_count: int
    frame_degenerate_count: int
    width_low_m: float = float("nan")
    width_high_m: float = float("nan")
    coordinate_frame: str = COORDINATE_FRAME_LEFT_RECTIFIED

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("Gripper side must be 'left' or 'right'.")
        if self.coordinate_frame != COORDINATE_FRAME_LEFT_RECTIFIED:
            raise ValueError("Gripper poses must stay in the left rectified camera.")
        count = _require_timeline(self.frame_indices, self.timestamps_ns, f"{self.side} gripper")
        _require_shape(self.position_m, (count, 3), f"{self.side}.position_m")
        _require_shape(self.quaternion_xyzw, (count, 4), f"{self.side}.quaternion_xyzw")
        _require_shape(self.opening_distance_m, (count,), f"{self.side}.opening_distance_m")
        _require_shape(self.opening_width, (count,), f"{self.side}.opening_width")
        _require_shape(self.invalid_mask, (count,), f"{self.side}.invalid_mask")


@dataclass(frozen=True)
class GripperPoses:
    """Both gripper poses on one shared timeline."""

    left: GripperPoseTrack
    right: GripperPoseTrack

    def __post_init__(self) -> None:
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("GripperPoses requires one left track and one right track.")
        if not np.array_equal(self.left.frame_indices, self.right.frame_indices):
            raise ValueError("Gripper poses must share frame indices.")
        if not np.array_equal(self.left.timestamps_ns, self.right.timestamps_ns):
            raise ValueError("Gripper poses must share timestamps.")


@dataclass(frozen=True)
class ArmAnchor:
    """One arm base chosen in the left rectified camera.

    An absent anchor keeps NaN pose fields and ``present`` false. A present
    anchor stores the camera-from-base translation in meters and a scipy xyzw
    quaternion. ``score`` is the Ego2Robot placement score.
    """

    side: str
    present: bool
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    score: float
    feasibility_rate: float
    mean_reach_fraction: float
    coordinate_frame: str = COORDINATE_FRAME_LEFT_RECTIFIED

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("Arm anchor side must be 'left' or 'right'.")
        if self.coordinate_frame != COORDINATE_FRAME_LEFT_RECTIFIED:
            raise ValueError("Arm anchors must stay in the left rectified camera.")
        _require_shape(self.position_m, (3,), f"{self.side}.anchor.position_m")
        _require_shape(self.quaternion_xyzw, (4,), f"{self.side}.anchor.quaternion_xyzw")
        if not self.present:
            return
        if not np.all(np.isfinite(self.position_m)) or not np.all(np.isfinite(self.quaternion_xyzw)):
            raise ValueError(f"{self.side} anchor pose must be finite when present.")
        if not np.isclose(np.linalg.norm(self.quaternion_xyzw), 1.0):
            raise ValueError(f"{self.side} anchor quaternion must be a unit quaternion.")
        if not np.isfinite(self.score):
            raise ValueError(f"{self.side} anchor score must be finite when present.")


@dataclass(frozen=True)
class ArmAnchors:
    """Independent left and right bases from one anchor search."""

    left: ArmAnchor
    right: ArmAnchor
    pair_score: float

    def __post_init__(self) -> None:
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("ArmAnchors requires one left anchor and one right anchor.")


@dataclass(frozen=True)
class TriangulatedHands:
    """Both anatomical hands after stereo triangulation, with no temporal repair."""

    left: HandTrack3D
    right: HandTrack3D

    def __post_init__(self) -> None:
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("TriangulatedHands requires one left track and one right track.")
        if not np.array_equal(self.left.frame_indices, self.right.frame_indices):
            raise ValueError("Left and right tracks must share frame indices.")
        if not np.array_equal(self.left.timestamps_ns, self.right.timestamps_ns):
            raise ValueError("Left and right tracks must share timestamps.")


def save_rectified_stereo(capture: RectifiedStereoForWiLoR, directory: Path) -> Path:
    """Write the preparation manifest. Timestamps stay int64 outside JSON.

    Args:
        capture: Rectified stereo bundle produced by preparation.
        directory: Run directory that already contains the rectified videos.
    Returns:
        Path of the JSON manifest.
    """

    directory.mkdir(parents=True, exist_ok=True)
    timeline_path = directory / "timeline.npz"
    manifest_path = directory / "capture.json"
    np.savez_compressed(
        timeline_path,
        frame_indices=np.asarray(capture.frame_indices, dtype=np.int64),
        timestamps_ns=np.asarray(capture.timestamps_ns, dtype=np.int64),
        is_synchronized=np.asarray(capture.is_synchronized, dtype=bool),
        sync_reason=np.asarray(capture.sync_reason, dtype="U64"),
    )
    payload = {
        "format": CAPTURE_FORMAT,
        "version": CAPTURE_VERSION,
        "session_id": capture.session_id,
        "pixel_space": capture.pixel_space,
        "timeline": timeline_path.name,
        "left_rectified_video": str(capture.left_rectified_video),
        "right_rectified_video": str(capture.right_rectified_video),
        "left_raw_video": str(capture.left_raw_video),
        "right_raw_video": str(capture.right_raw_video),
        "left_pinhole": _camera_payload(capture.left_pinhole),
        "right_pinhole": _camera_payload(capture.right_pinhole),
        "left_raw_fisheye": _camera_payload(capture.left_raw_fisheye),
        "right_raw_fisheye": _camera_payload(capture.right_raw_fisheye),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def load_rectified_stereo(directory: Path) -> RectifiedStereoForWiLoR:
    """Load a preparation manifest written by ``save_rectified_stereo``.

    Args:
        directory: Directory containing ``capture.json`` and ``timeline.npz``.
    Returns:
        The rectified stereo bundle, with timestamp precision preserved.
    Raises:
        ValueError: If the manifest format is not this pipeline's capture.
    """

    manifest_path = directory / "capture.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != CAPTURE_FORMAT or payload.get("version") != CAPTURE_VERSION:
        raise ValueError(f"Unsupported capture manifest: {manifest_path}")
    timeline = np.load(directory / payload["timeline"])
    return RectifiedStereoForWiLoR(
        session_id=str(payload["session_id"]),
        frame_indices=np.asarray(timeline["frame_indices"], dtype=np.int64),
        timestamps_ns=np.asarray(timeline["timestamps_ns"], dtype=np.int64),
        is_synchronized=np.asarray(timeline["is_synchronized"], dtype=bool),
        sync_reason=np.asarray(timeline["sync_reason"], dtype="U64"),
        left_rectified_video=Path(payload["left_rectified_video"]),
        right_rectified_video=Path(payload["right_rectified_video"]),
        left_pinhole=_camera_from_payload(payload["left_pinhole"]),
        right_pinhole=_camera_from_payload(payload["right_pinhole"]),
        left_raw_video=Path(payload["left_raw_video"]),
        right_raw_video=Path(payload["right_raw_video"]),
        left_raw_fisheye=_camera_from_payload(payload["left_raw_fisheye"]),
        right_raw_fisheye=_camera_from_payload(payload["right_raw_fisheye"]),
        pixel_space=str(payload["pixel_space"]),
    )


def save_camera_keypoints(keypoints: CameraKeypoints21, path: Path) -> Path:
    """Serialize one camera's WiLoR observations, including audit-only model 3D."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        camera_id=np.array(keypoints.camera_id),
        pixel_space=np.array(keypoints.pixel_space),
        backend_name=np.array(keypoints.backend_name),
        frame_indices=np.asarray(keypoints.frame_indices, dtype=np.int64),
        timestamps_ns=np.asarray(keypoints.timestamps_ns, dtype=np.int64),
        points_px=np.asarray(keypoints.points_px, dtype=np.float64),
        keypoint_confidence=np.asarray(keypoints.keypoint_confidence, dtype=np.float64),
        detection_confidence=np.asarray(keypoints.detection_confidence, dtype=np.float64),
        handedness=np.asarray(keypoints.handedness, dtype="U16"),
        handedness_confidence=np.asarray(keypoints.handedness_confidence, dtype=np.float64),
        hand_present=np.asarray(keypoints.hand_present, dtype=bool),
        status=np.asarray(keypoints.status, dtype="U32"),
        status_reason=np.asarray(keypoints.status_reason, dtype="U128"),
        model_relative_3d_m=np.asarray(keypoints.model_relative_3d_m, dtype=np.float64),
    )
    return path


def load_camera_keypoints(path: Path) -> CameraKeypoints21:
    """Load one ``CameraKeypoints21`` archive."""

    data = np.load(path)
    return CameraKeypoints21(
        camera_id=str(data["camera_id"].item()),
        pixel_space=str(data["pixel_space"].item()),
        frame_indices=np.asarray(data["frame_indices"], dtype=np.int64),
        timestamps_ns=np.asarray(data["timestamps_ns"], dtype=np.int64),
        points_px=np.asarray(data["points_px"], dtype=np.float64),
        keypoint_confidence=np.asarray(data["keypoint_confidence"], dtype=np.float64),
        detection_confidence=np.asarray(data["detection_confidence"], dtype=np.float64),
        handedness=np.asarray(data["handedness"], dtype="U16"),
        handedness_confidence=np.asarray(data["handedness_confidence"], dtype=np.float64),
        hand_present=np.asarray(data["hand_present"], dtype=bool),
        status=np.asarray(data["status"], dtype="U32"),
        status_reason=np.asarray(data["status_reason"], dtype="U128"),
        model_relative_3d_m=np.asarray(data["model_relative_3d_m"], dtype=np.float64),
        backend_name=str(data["backend_name"].item()),
    )


def save_triangulated_hands(result: TriangulatedHands, path: Path) -> Path:
    """Serialize triangulated tracks without smoothing or filling gaps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "frame_indices": np.asarray(result.left.frame_indices, dtype=np.int64),
        "timestamps_ns": np.asarray(result.left.timestamps_ns, dtype=np.int64),
    }
    for name, track in (("left", result.left), ("right", result.right)):
        payload[f"{name}_points_m"] = np.asarray(track.points_m, dtype=np.float64)
        payload[f"{name}_point_valid_mask"] = np.asarray(track.point_valid_mask, dtype=bool)
        payload[f"{name}_reprojection_error_px"] = np.asarray(track.reprojection_error_px, dtype=np.float64)
        payload[f"{name}_frame_status"] = np.asarray(track.frame_status, dtype="U32")
        payload[f"{name}_failure_reason"] = np.asarray(track.failure_reason, dtype="U128")
    np.savez_compressed(path, **payload)
    return path


def load_triangulated_hands(path: Path) -> TriangulatedHands:
    """Load both hands saved by ``save_triangulated_hands``."""

    data = np.load(path)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    timestamps_ns = np.asarray(data["timestamps_ns"], dtype=np.int64)
    return TriangulatedHands(
        left=_load_raw("left", data, frame_indices, timestamps_ns),
        right=_load_raw("right", data, frame_indices, timestamps_ns),
    )


def save_repaired_hands(result: RepairedHands, path: Path) -> Path:
    """Serialize repaired tracks without changing the raw triangulation file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "frame_indices": np.asarray(result.left.raw.frame_indices, dtype=np.int64),
        "timestamps_ns": np.asarray(result.left.raw.timestamps_ns, dtype=np.int64),
    }
    for name, track in (("left", result.left), ("right", result.right)):
        raw = track.raw
        payload[f"{name}_raw_points_m"] = np.asarray(raw.points_m, dtype=np.float64)
        payload[f"{name}_raw_point_valid_mask"] = np.asarray(raw.point_valid_mask, dtype=bool)
        payload[f"{name}_reprojection_error_px"] = np.asarray(raw.reprojection_error_px, dtype=np.float64)
        payload[f"{name}_frame_status"] = np.asarray(raw.frame_status, dtype="U32")
        payload[f"{name}_failure_reason"] = np.asarray(raw.failure_reason, dtype="U128")
        payload[f"{name}_points_m"] = np.asarray(track.points_m, dtype=np.float64)
        payload[f"{name}_invalid_mask"] = np.asarray(track.invalid_mask, dtype=bool)
    np.savez_compressed(path, **payload)
    return path


def load_repaired_hands(path: Path) -> RepairedHands:
    """Load both hands saved by ``save_repaired_hands``."""

    data = np.load(path)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    timestamps_ns = np.asarray(data["timestamps_ns"], dtype=np.int64)
    return RepairedHands(
        left=_load_repaired("left", data, frame_indices, timestamps_ns),
        right=_load_repaired("right", data, frame_indices, timestamps_ns),
    )


def save_gripper_poses(result: GripperPoses, path: Path) -> Path:
    """Serialize both gripper poses without changing the source keypoints."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "frame_indices": np.asarray(result.left.frame_indices, dtype=np.int64),
        "timestamps_ns": np.asarray(result.left.timestamps_ns, dtype=np.int64),
    }
    for name, track in (("left", result.left), ("right", result.right)):
        payload[f"{name}_position_m"] = np.asarray(track.position_m, dtype=np.float64)
        payload[f"{name}_quaternion_xyzw"] = np.asarray(track.quaternion_xyzw, dtype=np.float64)
        payload[f"{name}_opening_distance_m"] = np.asarray(track.opening_distance_m, dtype=np.float64)
        payload[f"{name}_opening_width"] = np.asarray(track.opening_width, dtype=np.float64)
        payload[f"{name}_invalid_mask"] = np.asarray(track.invalid_mask, dtype=bool)
        payload[f"{name}_jaw_degenerate_count"] = np.asarray(track.jaw_degenerate_count, dtype=np.int64)
        payload[f"{name}_frame_degenerate_count"] = np.asarray(track.frame_degenerate_count, dtype=np.int64)
        payload[f"{name}_width_low_m"] = np.asarray(track.width_low_m, dtype=np.float64)
        payload[f"{name}_width_high_m"] = np.asarray(track.width_high_m, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def save_arm_anchors(result: ArmAnchors, path: Path) -> Path:
    """Serialize both arm bases without changing the gripper trajectory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"pair_score": np.asarray(result.pair_score, dtype=np.float64)}
    for name, anchor in (("left", result.left), ("right", result.right)):
        payload[f"{name}_present"] = np.asarray(anchor.present, dtype=bool)
        payload[f"{name}_position_m"] = np.asarray(anchor.position_m, dtype=np.float64)
        payload[f"{name}_quaternion_xyzw"] = np.asarray(anchor.quaternion_xyzw, dtype=np.float64)
        payload[f"{name}_score"] = np.asarray(anchor.score, dtype=np.float64)
        payload[f"{name}_feasibility_rate"] = np.asarray(anchor.feasibility_rate, dtype=np.float64)
        payload[f"{name}_mean_reach_fraction"] = np.asarray(anchor.mean_reach_fraction, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def load_arm_anchors(path: Path) -> ArmAnchors:
    """Load both arm bases saved by ``save_arm_anchors``."""

    data = np.load(path)
    return ArmAnchors(
        left=_load_anchor("left", data),
        right=_load_anchor("right", data),
        pair_score=float(data["pair_score"]),
    )


def load_gripper_poses(path: Path) -> GripperPoses:
    """Load both gripper poses saved by ``save_gripper_poses``."""

    data = np.load(path)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    timestamps_ns = np.asarray(data["timestamps_ns"], dtype=np.int64)
    return GripperPoses(
        left=_load_gripper("left", data, frame_indices, timestamps_ns),
        right=_load_gripper("right", data, frame_indices, timestamps_ns),
    )


def _load_anchor(side: str, data: np.lib.npyio.NpzFile) -> ArmAnchor:
    return ArmAnchor(
        side=side,
        present=bool(data[f"{side}_present"]),
        position_m=np.asarray(data[f"{side}_position_m"], dtype=np.float64),
        quaternion_xyzw=np.asarray(data[f"{side}_quaternion_xyzw"], dtype=np.float64),
        score=float(data[f"{side}_score"]),
        feasibility_rate=float(data[f"{side}_feasibility_rate"]),
        mean_reach_fraction=float(data[f"{side}_mean_reach_fraction"]),
    )


def _load_gripper(
    side: str, data: np.lib.npyio.NpzFile, frame_indices: np.ndarray, timestamps_ns: np.ndarray
) -> GripperPoseTrack:
    return GripperPoseTrack(
        side=side,
        frame_indices=frame_indices,
        timestamps_ns=timestamps_ns,
        position_m=np.asarray(data[f"{side}_position_m"], dtype=np.float64),
        quaternion_xyzw=np.asarray(data[f"{side}_quaternion_xyzw"], dtype=np.float64),
        opening_distance_m=np.asarray(data[f"{side}_opening_distance_m"], dtype=np.float64),
        opening_width=np.asarray(data[f"{side}_opening_width"], dtype=np.float64),
        invalid_mask=np.asarray(data[f"{side}_invalid_mask"], dtype=bool),
        jaw_degenerate_count=int(data[f"{side}_jaw_degenerate_count"]),
        frame_degenerate_count=int(data[f"{side}_frame_degenerate_count"]),
        width_low_m=float(data[f"{side}_width_low_m"]),
        width_high_m=float(data[f"{side}_width_high_m"]),
    )


def _load_repaired(
    side: str, data: np.lib.npyio.NpzFile, frame_indices: np.ndarray, timestamps_ns: np.ndarray
) -> RepairedHandTrack:
    raw = HandTrack3D(
        side=side,
        frame_indices=frame_indices,
        timestamps_ns=timestamps_ns,
        points_m=np.asarray(data[f"{side}_raw_points_m"], dtype=np.float64),
        point_valid_mask=np.asarray(data[f"{side}_raw_point_valid_mask"], dtype=bool),
        reprojection_error_px=np.asarray(data[f"{side}_reprojection_error_px"], dtype=np.float64),
        frame_status=np.asarray(data[f"{side}_frame_status"], dtype="U32"),
        failure_reason=np.asarray(data[f"{side}_failure_reason"], dtype="U128"),
    )
    return RepairedHandTrack(
        raw=raw,
        points_m=np.asarray(data[f"{side}_points_m"], dtype=np.float64),
        invalid_mask=np.asarray(data[f"{side}_invalid_mask"], dtype=bool),
    )


def _load_raw(side: str, data: np.lib.npyio.NpzFile, frame_indices: np.ndarray, timestamps_ns: np.ndarray) -> HandTrack3D:
    return HandTrack3D(
        side=side,
        frame_indices=frame_indices,
        timestamps_ns=timestamps_ns,
        points_m=np.asarray(data[f"{side}_points_m"], dtype=np.float64),
        point_valid_mask=np.asarray(data[f"{side}_point_valid_mask"], dtype=bool),
        reprojection_error_px=np.asarray(data[f"{side}_reprojection_error_px"], dtype=np.float64),
        frame_status=np.asarray(data[f"{side}_frame_status"], dtype="U32"),
        failure_reason=np.asarray(data[f"{side}_failure_reason"], dtype="U128"),
    )


def _camera_payload(camera: CameraModel) -> dict[str, object]:
    return {
        "camera_id": camera.camera_id,
        "camera_model": camera.camera_model,
        "image_size_px": [camera.image_size_px[0], camera.image_size_px[1]],
        "intrinsics": [list(row) for row in camera.intrinsics],
        "distortion": list(camera.distortion),
        "T_rig_from_camera": [list(row) for row in camera.T_rig_from_camera],
    }


def _camera_from_payload(payload: dict[str, object]) -> CameraModel:
    intrinsics = tuple(tuple(float(value) for value in row) for row in payload["intrinsics"])  # type: ignore[index]
    transform = tuple(tuple(float(value) for value in row) for row in payload["T_rig_from_camera"])  # type: ignore[index]
    image_size = payload["image_size_px"]
    return CameraModel(
        camera_id=str(payload["camera_id"]),
        camera_model=str(payload["camera_model"]),
        image_size_px=(int(image_size[0]), int(image_size[1])),  # type: ignore[index]
        intrinsics=intrinsics,  # type: ignore[arg-type]
        distortion=tuple(float(value) for value in payload["distortion"]),  # type: ignore[union-attr]
        T_rig_from_camera=transform,  # type: ignore[arg-type]
    )


def _require_timeline(frame_indices: np.ndarray, timestamps_ns: np.ndarray, label: str) -> int:
    if frame_indices.ndim != 1 or timestamps_ns.shape != frame_indices.shape:
        raise ValueError(f"{label} frame indices and timestamps must share one length.")
    if len(frame_indices) == 0:
        raise ValueError(f"{label} timeline must not be empty.")
    if np.any(timestamps_ns[1:] <= timestamps_ns[:-1]):
        raise ValueError(f"{label} timestamps must be strictly increasing.")
    return int(len(frame_indices))


def _require_shape(value: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if np.shape(value) != shape:
        raise ValueError(f"{label} must have shape {shape}, received {np.shape(value)}.")
