"""Module 4: triangulate smoothed rectified pixels with the SSH pinhole measurement.

Detections already live on rectified pinhole frames, so this follows
``triangulate_hand_slot(..., pre_undistorted=True)``. Points are meters in the
left rectified camera. Invalid points are zeros. Wrist-relative WiLoR 3D is
not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts import (
    KEYPOINT_COUNT,
    COORDINATE_FRAME_LEFT_RECTIFIED,
    CameraKeypoints21,
    CameraModel,
    HandTrack3D,
    RectifiedStereoForWiLoR,
    TriangulatedHands,
)


@dataclass(frozen=True)
class TriangulationConfig:
    """SSH reprojection gate for an already-rectified stereo pair.

    Args:
        maximum_reprojection_error_px: A point is invalid when either view
            exceeds this. SSH's pipeline default is ``10`` pixels.
    """

    maximum_reprojection_error_px: float = 10.0

    def __post_init__(self) -> None:
        if self.maximum_reprojection_error_px <= 0.0:
            raise ValueError("maximum_reprojection_error_px must be positive.")


def triangulate_smoothed_keypoints(
    capture: RectifiedStereoForWiLoR,
    left: CameraKeypoints21,
    right: CameraKeypoints21,
    config: TriangulationConfig,
) -> TriangulatedHands:
    """Triangulate anatomical hands that module 3 already aligned by side.

    Args:
        capture: Rectified pinhole models. Their poses include the rectification rotation.
        left: Smoothed left-camera observations. Slot 0 is the left hand.
        right: Smoothed right-camera observations. Slot 0 is the left hand.
        config: Reprojection threshold.
    Returns:
        Both hands in the left rectified camera, meters. A point is valid only
        when both views saw that hand, both reprojection errors pass, and
        camera ``z`` is positive.
    """

    _require_same_timeline(capture, left, right)
    projection_left, projection_right = _projection_matrices(capture.left_pinhole, capture.right_pinhole)
    tracks = []
    for slot, side in enumerate(("left", "right")):
        points_m, valid, error_px, status, reason = _triangulate_slot(
            left, right, slot, projection_left, projection_right, config.maximum_reprojection_error_px
        )
        tracks.append(
            HandTrack3D(
                side=side,
                frame_indices=np.asarray(left.frame_indices, dtype=np.int64),
                timestamps_ns=np.asarray(left.timestamps_ns, dtype=np.int64),
                points_m=points_m,
                point_valid_mask=valid,
                reprojection_error_px=error_px,
                frame_status=status,
                failure_reason=reason,
                coordinate_frame=COORDINATE_FRAME_LEFT_RECTIFIED,
            )
        )
    return TriangulatedHands(left=tracks[0], right=tracks[1])


def _triangulate_slot(
    left: CameraKeypoints21,
    right: CameraKeypoints21,
    slot: int,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    maximum_reprojection_error_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copy ``triangulate_hand_slot`` for one anatomical hand."""

    frame_count = len(left.frame_indices)
    points_m = np.zeros((frame_count, KEYPOINT_COUNT, 3), dtype=np.float64)
    valid = np.zeros((frame_count, KEYPOINT_COUNT), dtype=bool)
    error_px = np.full((frame_count, KEYPOINT_COUNT), np.nan, dtype=np.float64)
    status = np.full(frame_count, "missing", dtype="U32")
    reason = np.full(frame_count, "hand_not_in_both_views", dtype="U128")
    both_present = left.hand_present[:, slot] & right.hand_present[:, slot]
    for frame_index in np.flatnonzero(both_present):
        uv_left = np.asarray(left.points_px[frame_index, slot], dtype=np.float64)
        uv_right = np.asarray(right.points_px[frame_index, slot], dtype=np.float64)
        points = _triangulate_points(projection_left, projection_right, uv_left, uv_right)
        error_left = _reprojection_error(projection_left, points, uv_left)
        error_right = _reprojection_error(projection_right, points, uv_right)
        accepted = (error_left < maximum_reprojection_error_px) & (error_right < maximum_reprojection_error_px) & (points[:, 2] > 0)
        points_m[frame_index] = np.where(accepted[:, None], points, 0.0)
        valid[frame_index] = accepted
        error_px[frame_index] = np.maximum(error_left, error_right)
        if accepted.any():
            status[frame_index] = "measured"
            reason[frame_index] = ""
        else:
            status[frame_index] = "rejected"
            reason[frame_index] = "reprojection_or_depth"
    return points_m, valid, error_px, status, reason


def _projection_matrices(left_camera: CameraModel, right_camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """Build SSH ``P1`` and ``P2`` from rig poses of the rectified cameras.

    ``T_rig_from_camera`` maps camera coordinates into the rig. The right
    projection therefore uses the transform that maps a left-camera point
    into the right camera, which is what SSH calls ``T_left_right``.
    """

    intrinsics_left = np.asarray(left_camera.intrinsics, dtype=np.float64)
    intrinsics_right = np.asarray(right_camera.intrinsics, dtype=np.float64)
    rig_from_left = np.asarray(left_camera.T_rig_from_camera, dtype=np.float64)
    rig_from_right = np.asarray(right_camera.T_rig_from_camera, dtype=np.float64)
    right_from_left = np.linalg.inv(rig_from_right) @ rig_from_left
    projection_left = intrinsics_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_right = intrinsics_right @ right_from_left[:3, :]
    return projection_left, projection_right


def _triangulate_points(projection_left: np.ndarray, projection_right: np.ndarray, uv_left: np.ndarray, uv_right: np.ndarray) -> np.ndarray:
    """SSH ``triangulate_points``: returns ``(21, 3)`` in the left camera."""

    import cv2

    homogeneous = cv2.triangulatePoints(projection_left, projection_right, uv_left.T, uv_right.T)
    return (homogeneous[:3] / homogeneous[3]).T


def _reprojection_error(projection: np.ndarray, points_m: np.ndarray, uv_observed: np.ndarray) -> np.ndarray:
    """SSH ``reprojection_error`` in pixels."""

    ones = np.ones((points_m.shape[0], 1))
    projected = np.concatenate([points_m, ones], axis=1) @ projection.T
    uv_projected = projected[:, :2] / projected[:, 2:3]
    return np.linalg.norm(uv_projected - uv_observed, axis=1)


def _require_same_timeline(capture: RectifiedStereoForWiLoR, left: CameraKeypoints21, right: CameraKeypoints21) -> None:
    if not np.array_equal(left.frame_indices, right.frame_indices) or not np.array_equal(left.frame_indices, capture.frame_indices):
        raise ValueError("Smoothed keypoints and the rectified capture must share frame indices.")
    if not np.array_equal(left.timestamps_ns, right.timestamps_ns):
        raise ValueError("Smoothed keypoints must share timestamps.")
