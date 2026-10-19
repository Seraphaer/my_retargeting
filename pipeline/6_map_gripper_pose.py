"""Module 6: turn filled 21-point tracks into a raw gripper pose.

The jaw-plane math lives in ``gripper_pose.jaw_geometry``. This file only
orders that calculation. It does not filter the finished trajectory.
"""

from __future__ import annotations

from contracts import GripperPoses, RepairedHands
from gripper_pose.jaw_geometry import JawPoseConfig, map_raw_gripper_poses


def map_gripper_poses(filled: RepairedHands, config: JawPoseConfig | None = None) -> GripperPoses:
    """Return the raw gripper pose for both hands.

    Args:
        filled: Module 5 output. Points are meters in the left rectified camera.
        config: Optional input-smoothing switches. ``None`` uses the SSH defaults.
    Returns:
        Positions, jaw-plane quaternions, and meter opening distances.
        Normalized opening width is still NaN.
    """

    return map_raw_gripper_poses(filled, JawPoseConfig() if config is None else config)
