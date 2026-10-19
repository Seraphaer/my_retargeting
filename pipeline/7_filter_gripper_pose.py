"""Module 7: apply named gripper-pose filters in a fixed order.

Each stage is one function. A stage omitted from ``enabled_stages`` passes
the previous pose through unchanged. This file contains no filter formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from contracts import GripperPoseTrack, GripperPoses
from gripper_pose.gap_blend import blend_long_gaps
from gripper_pose.pose_filters import (
    gaussian_slerp_smooth,
    one_euro_filter,
    orientation_velocity_outlier_filter,
    savgol_smooth,
    velocity_outlier_filter,
)


STAGE_ORDER = (
    "long_gap_blend",
    "width_one_euro",
    "position_velocity_outlier",
    "position_savgol",
    "orientation_velocity_outlier",
    "orientation_gaussian_slerp",
)


@dataclass(frozen=True)
class GripperFilterConfig:
    """Which pose stages run, and the frame rate those stages share.

    Args:
        frames_per_second: Rectified-video rate used by One-Euro and the velocity gates.
        enabled_stages: Stage names in any order. They still run in ``STAGE_ORDER``.
            An empty tuple leaves the module 6 pose unchanged.
        long_gap_frames: Upper length of a gap that ``long_gap_blend`` ignores.
    """

    frames_per_second: float
    enabled_stages: tuple[str, ...] = STAGE_ORDER
    long_gap_frames: int = 10

    def __post_init__(self) -> None:
        if self.frames_per_second <= 0.0:
            raise ValueError("frames_per_second must be positive.")
        unknown = set(self.enabled_stages) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"Unknown gripper filter stages: {sorted(unknown)}")


def filter_gripper_poses(poses: GripperPoses, config: GripperFilterConfig) -> list[tuple[str, GripperPoses]]:
    """Run the enabled stages and return one snapshot after each of them.

    Args:
        poses: Raw poses from module 6.
        config: Frame rate and the stage names to apply.
    Returns:
        ``(stage_name, poses)`` pairs for enabled stages only. An empty config
        returns an empty list; the caller keeps the raw pose.
    """

    snapshots: list[tuple[str, GripperPoses]] = []
    current = poses
    enabled = set(config.enabled_stages)
    for stage_name in STAGE_ORDER:
        if stage_name not in enabled:
            continue
        current = GripperPoses(
            left=_apply_stage(current.left, stage_name, config),
            right=_apply_stage(current.right, stage_name, config),
        )
        snapshots.append((stage_name, current))
    return snapshots


def _apply_stage(track: GripperPoseTrack, stage_name: str, config: GripperFilterConfig) -> GripperPoseTrack:
    if stage_name == "long_gap_blend":
        return blend_long_gaps(track, config.long_gap_frames)
    if stage_name == "width_one_euro":
        return _normalize_and_filter_width(track, config.frames_per_second)
    if stage_name == "position_velocity_outlier":
        return replace(track, position_m=velocity_outlier_filter(track.position_m, config.frames_per_second))
    if stage_name == "position_savgol":
        return replace(track, position_m=savgol_smooth(track.position_m))
    if stage_name == "orientation_velocity_outlier":
        return replace(track, quaternion_xyzw=orientation_velocity_outlier_filter(track.quaternion_xyzw, config.frames_per_second))
    if stage_name == "orientation_gaussian_slerp":
        return replace(track, quaternion_xyzw=gaussian_slerp_smooth(track.quaternion_xyzw))
    raise ValueError(f"Unknown gripper filter stage: {stage_name}")


def _normalize_and_filter_width(track: GripperPoseTrack, frames_per_second: float) -> GripperPoseTrack:
    valid = ~np.asarray(track.invalid_mask, dtype=bool)
    distances = np.asarray(track.opening_distance_m, dtype=float)
    if valid.any():
        low_m = float(np.percentile(distances[valid], 2))
        high_m = float(np.percentile(distances[valid], 98))
    else:
        low_m, high_m = 0.0, 1.0
    span_m = high_m - low_m
    if span_m <= 1e-8:
        normalized = np.zeros_like(distances)
    else:
        normalized = np.clip((distances - low_m) / span_m, 0.0, 1.0)
    filtered = np.clip(one_euro_filter(normalized, frames_per_second, minimum_cutoff_hz=1.0, beta=15.0), 0.0, 1.0)
    return replace(track, opening_width=filtered, width_low_m=low_m, width_high_m=high_m)
