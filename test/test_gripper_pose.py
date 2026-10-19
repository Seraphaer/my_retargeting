"""Synthetic checks for the raw jaw pose and for disabling module 7."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from contracts import HandTrack3D, RepairedHandTrack, RepairedHands
from importlib import import_module

_gripper = import_module("6_map_gripper_pose")
_filter = import_module("7_filter_gripper_pose")


def test_pinch_pose_is_a_right_handed_frame() -> None:
    """A separated thumb and index produce a finite midpoint and a unit quaternion."""

    poses = _gripper.map_gripper_poses(_pinch_hands(), _gripper.JawPoseConfig(smooth_closing_inputs=False, smooth_reach_inputs=False))
    position = poses.left.position_m[0]
    quaternion = poses.left.quaternion_xyzw[0]
    assert np.allclose(position, [0.0, 0.02, 0.4])
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    assert np.isclose(np.linalg.det(Rotation.from_quat(quaternion).as_matrix()), 1.0)
    assert np.isclose(poses.left.opening_distance_m[0], 0.04)
    assert np.isnan(poses.left.opening_width).all()


def test_disabling_every_filter_keeps_the_raw_pose() -> None:
    """Module 7 with an empty stage list must not change module 6."""

    raw = _gripper.map_gripper_poses(_pinch_hands())
    snapshots = _filter.filter_gripper_poses(raw, _filter.GripperFilterConfig(frames_per_second=30.0, enabled_stages=()))
    assert snapshots == []


def _pinch_hands() -> RepairedHands:
    points = np.zeros((8, 21, 3))
    points[:, 0] = [0.0, 0.0, 0.4]
    points[:, 4] = [-0.02, 0.02, 0.4]
    points[:, 8] = [0.02, 0.02, 0.4]
    points[:, 12] = [0.0, 0.05, 0.42]
    points[:, 9] = [0.0, 0.03, 0.4]
    frame_indices = np.arange(8)
    timestamps = frame_indices * 33_000_000

    def track(side: str) -> RepairedHandTrack:
        raw = HandTrack3D(
            side=side,
            frame_indices=frame_indices,
            timestamps_ns=timestamps,
            points_m=points,
            point_valid_mask=np.ones((8, 21), dtype=bool),
            reprojection_error_px=np.zeros((8, 21)),
            frame_status=np.full(8, "measured", dtype="U32"),
            failure_reason=np.full(8, "", dtype="U128"),
        )
        return RepairedHandTrack(raw=raw, points_m=points, invalid_mask=np.zeros(8, dtype=bool))

    return RepairedHands(left=track("left"), right=track("right"))


if __name__ == "__main__":
    test_pinch_pose_is_a_right_handed_frame()
    test_disabling_every_filter_keeps_the_raw_pose()
    print("gripper pose checks ok")
