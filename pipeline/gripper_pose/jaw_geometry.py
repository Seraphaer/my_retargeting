"""Thumb-index jaw-plane gripper pose from filled 21-point tracks.

This file defines the raw pose. Trajectory filters that run after the pose
exists belong to module 7. No SSH package is imported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from contracts import GripperPoseTrack, GripperPoses, RepairedHandTrack, RepairedHands
from gripper_pose.pose_filters import ema_smooth, hampel_filter, normalize_rows


WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12


@dataclass(frozen=True)
class JawPoseConfig:
    """Switches and thresholds for the raw thumb-index jaw pose.

    Args:
        smooth_closing_inputs: EMA the thumb and index tips before the jaw line.
        smooth_reach_inputs: Hampel and EMA the wrist and fingertip before the reach axis.
        closing_ema_alpha: Fixed EMA weight for the jaw-line inputs.
        reach_ema_alpha: Fixed EMA weight for the reach-axis inputs.
        reach_hampel_window: Neighborhood radius, in frames, for the reach Hampel filter.
        reach_hampel_n_sigma: Robust-deviation multiplier that replaces a reach outlier.
        tau_close_m: Below this pinch distance, the jaw line blends toward wrist-to-index.
        jaw_degenerate_sin: Freeze the jaw normal when the plane's sine falls below this.
        pinch_anchor_ratio: Fraction from thumb tip toward index tip used as the origin.
        virtual_index_weight: Index-tip weight of the virtual fingertip.
        virtual_middle_weight: Middle-tip weight of the virtual fingertip.
    """

    smooth_closing_inputs: bool = True
    smooth_reach_inputs: bool = True
    closing_ema_alpha: float = 0.1
    reach_ema_alpha: float = 0.15
    reach_hampel_window: int = 5
    reach_hampel_n_sigma: float = 3.0
    tau_close_m: float = 0.01
    jaw_degenerate_sin: float = 0.15
    pinch_anchor_ratio: float = 0.5
    virtual_index_weight: float = 0.7
    virtual_middle_weight: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 < self.tau_close_m:
            raise ValueError("tau_close_m must be positive.")
        if not 0.0 <= self.pinch_anchor_ratio <= 1.0:
            raise ValueError("pinch_anchor_ratio must be inside [0, 1].")


def map_raw_gripper_poses(filled: RepairedHands, config: JawPoseConfig) -> GripperPoses:
    """Compute one raw gripper pose for each filled hand.

    Args:
        filled: Module 5 tracks. Points are meters in the left rectified camera.
        config: Jaw-plane thresholds and the two optional input smoothers.
    Returns:
        Raw poses. Opening width is NaN because normalization belongs to module 7.
    """

    return GripperPoses(
        left=_pose_from_hand(filled.left, config),
        right=_pose_from_hand(filled.right, config),
    )


def _pose_from_hand(track: RepairedHandTrack, config: JawPoseConfig) -> GripperPoseTrack:
    points = np.asarray(track.points_m, dtype=float)
    thumb = points[:, THUMB_TIP]
    index = points[:, INDEX_TIP]
    opening = np.linalg.norm(index - thumb, axis=-1)
    position = thumb + config.pinch_anchor_ratio * (index - thumb)
    closing_thumb, closing_index = _closing_points(points, config)
    closing = _closing_direction(points, closing_thumb, closing_index, opening, config)
    reach = _reach_direction(points, config)
    quaternion, jaw_count, frame_count = _jaw_quaternion(closing, reach, config)
    frame_count_frames = len(points)
    return GripperPoseTrack(
        side=track.raw.side,
        frame_indices=np.asarray(track.raw.frame_indices, dtype=np.int64),
        timestamps_ns=np.asarray(track.raw.timestamps_ns, dtype=np.int64),
        position_m=position,
        quaternion_xyzw=quaternion,
        opening_distance_m=opening,
        opening_width=np.full(frame_count_frames, np.nan),
        invalid_mask=np.asarray(track.invalid_mask, dtype=bool),
        jaw_degenerate_count=jaw_count,
        frame_degenerate_count=frame_count,
    )


def _closing_points(points: np.ndarray, config: JawPoseConfig) -> tuple[np.ndarray, np.ndarray]:
    thumb = points[:, THUMB_TIP]
    index = points[:, INDEX_TIP]
    if not config.smooth_closing_inputs:
        return thumb, index
    return ema_smooth(thumb, config.closing_ema_alpha), ema_smooth(index, config.closing_ema_alpha)


def _closing_direction(
    points: np.ndarray, thumb: np.ndarray, index: np.ndarray, opening_m: np.ndarray, config: JawPoseConfig
) -> np.ndarray:
    closing = normalize_rows(index - thumb)
    fallback = normalize_rows(points[:, INDEX_TIP] - points[:, WRIST])
    blend = np.clip(opening_m / config.tau_close_m, 0.0, 1.0)[:, None]
    return normalize_rows(blend * closing + (1.0 - blend) * fallback)


def _reach_direction(points: np.ndarray, config: JawPoseConfig) -> np.ndarray:
    weight_sum = config.virtual_index_weight + config.virtual_middle_weight
    index_weight = config.virtual_index_weight / weight_sum
    middle_weight = config.virtual_middle_weight / weight_sum
    if config.smooth_reach_inputs:
        wrist = _smooth_reach_column(points[:, WRIST], config)
        index = _smooth_reach_column(points[:, INDEX_TIP], config)
        middle = _smooth_reach_column(points[:, MIDDLE_TIP], config)
    else:
        wrist = points[:, WRIST]
        index = points[:, INDEX_TIP]
        middle = points[:, MIDDLE_TIP]
    fingertip = index_weight * index + middle_weight * middle
    return normalize_rows(fingertip - wrist)


def _smooth_reach_column(column: np.ndarray, config: JawPoseConfig) -> np.ndarray:
    filtered = hampel_filter(column, config.reach_hampel_window, config.reach_hampel_n_sigma)
    return ema_smooth(filtered, config.reach_ema_alpha)


def _jaw_quaternion(closing: np.ndarray, reach: np.ndarray, config: JawPoseConfig) -> tuple[np.ndarray, int, int]:
    cross_jaw = np.cross(reach, closing)
    sine = np.linalg.norm(cross_jaw, axis=-1) / np.maximum(
        np.linalg.norm(reach, axis=-1) * np.linalg.norm(closing, axis=-1), 1e-12
    )
    jaw_degenerate = sine < config.jaw_degenerate_sin
    normal = _sign_stabilize(_forward_fill(normalize_rows(cross_jaw), jaw_degenerate), jaw_degenerate)
    approach = normalize_rows(np.cross(closing, normal))
    rotation = np.stack([closing, normal, approach], axis=-1)
    frame_sine = np.linalg.norm(np.cross(closing, normal), axis=-1) / np.maximum(
        np.linalg.norm(closing, axis=-1) * np.linalg.norm(normal, axis=-1), 1e-12
    )
    frame_degenerate = frame_sine < config.jaw_degenerate_sin
    rotation = _forward_fill(rotation, frame_degenerate)
    quaternion = Rotation.from_matrix(rotation).as_quat()
    quaternion = _ramp_after_freezes(quaternion, jaw_degenerate | frame_degenerate)
    return quaternion, int(jaw_degenerate.sum()), int(frame_degenerate.sum())


def _sign_stabilize(normal: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    """Keep the jaw normal's sign continuous inside each run of live frames."""

    sign = np.ones(len(normal))
    segment_sign = 1.0
    carried_sign = 1.0
    previous_was_live = False
    for frame_index in range(len(normal)):
        if frozen[frame_index]:
            sign[frame_index] = carried_sign
            previous_was_live = False
            continue
        if not previous_was_live:
            segment_sign = 1.0
            sign[frame_index] = 1.0
        else:
            if np.dot(normal[frame_index], normal[frame_index - 1]) < 0.0:
                segment_sign = -segment_sign
            sign[frame_index] = segment_sign
        carried_sign = sign[frame_index]
        previous_was_live = True
    return normal * sign[:, None]


def _forward_fill(values: np.ndarray, degenerate: np.ndarray) -> np.ndarray:
    valid_index = np.flatnonzero(~degenerate)
    if len(valid_index) == 0:
        return values.copy()
    output = values.copy()
    first_valid = int(valid_index[0])
    if first_valid > 0:
        output[:first_valid] = values[first_valid]
    last_valid = values[first_valid]
    for frame_index in range(first_valid, len(degenerate)):
        if degenerate[frame_index]:
            output[frame_index] = last_valid
        else:
            last_valid = values[frame_index]
    return output


def _ramp_after_freezes(
    quaternions: np.ndarray, degenerate: np.ndarray, degrees_per_frame: float = 8.0, minimum_degrees: float = 3.0, maximum_frames: int = 20
) -> np.ndarray:
    """SLERP from a frozen orientation into the next live orientation."""

    output = quaternions.copy()
    segments = _true_segments(degenerate)
    frame_count = len(degenerate)
    for segment_index, (start, end, length) in enumerate(segments):
        if end >= frame_count - 1:
            continue
        next_start = segments[segment_index + 1][0] if segment_index + 1 < len(segments) else frame_count
        held = output[end - 1]
        gap_degrees = float(np.degrees(2.0 * np.arccos(np.clip(abs(np.dot(held, output[end])), -1.0, 1.0))))
        if gap_degrees < minimum_degrees:
            continue
        ramp_frames = int(np.clip(np.ceil(gap_degrees / degrees_per_frame), 2, maximum_frames))
        forward_room = min(frame_count - 1 - end, next_start - end - 1)
        backward_room = length - 1
        ramp_frames = min(ramp_frames, forward_room + backward_room)
        if ramp_frames <= 1:
            continue
        forward_used = min(forward_room, ramp_frames)
        backward_used = ramp_frames - forward_used
        window_start = end - backward_used
        window_end = end + forward_used
        blend = Slerp([0.0, 1.0], Rotation.from_quat([held, output[window_end]]))
        for offset, frame_index in enumerate(range(window_start, window_end)):
            fraction = _smoothstep((offset + 1) / ramp_frames)
            output[frame_index] = blend([fraction]).as_quat()[0]
    return output


def _true_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    frame_index = 0
    while frame_index < len(mask):
        if not mask[frame_index]:
            frame_index += 1
            continue
        start = frame_index
        while frame_index < len(mask) and mask[frame_index]:
            frame_index += 1
        segments.append((start, frame_index, frame_index - start))
    return segments


def _smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)
