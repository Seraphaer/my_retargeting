"""Blend the tail of a long keypoint gap into the next measured gripper pose.

The invalid mask comes from module 5. This file does not detect gaps or
change the 21 keypoints.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from contracts import GripperPoseTrack


def blend_long_gaps(track: GripperPoseTrack, long_gap_frames: int = 10) -> GripperPoseTrack:
    """Blend position, opening distance, and orientation across each long gap's tail.

    Args:
        track: Raw or partially filtered gripper pose. Its invalid mask marks
            frames that were missing before keypoint fill.
        long_gap_frames: Gaps no longer than this are left unchanged.
    Returns:
        A new pose. Leading and trailing gaps are unchanged.
    """

    position = np.array(track.position_m, copy=True)
    opening = np.array(track.opening_distance_m, copy=True)
    quaternion = np.array(track.quaternion_xyzw, copy=True)
    invalid = np.asarray(track.invalid_mask, dtype=bool)
    frame_count = len(invalid)
    frame_index = 0
    while frame_index < frame_count:
        if not invalid[frame_index]:
            frame_index += 1
            continue
        start = frame_index
        while frame_index < frame_count and invalid[frame_index]:
            frame_index += 1
        length = frame_index - start
        if length <= long_gap_frames or start == 0 or frame_index >= frame_count:
            continue
        before = start - 1
        after = frame_index
        displacement_m = float(np.linalg.norm(position[after] - position[before]))
        angle_degrees = float(np.degrees(2.0 * np.arccos(np.clip(abs(np.dot(quaternion[before], quaternion[after])), -1.0, 1.0))))
        blend_frames = int(np.clip(np.ceil(0.6 * displacement_m / 0.00325 + 0.4 * angle_degrees / 1.08), 5, 90))
        blend_frames = min(blend_frames, length)
        tail_start = frame_index - blend_frames
        orientation = Slerp([0.0, 1.0], Rotation.from_quat([quaternion[before], quaternion[after]]))
        for offset, sample_index in enumerate(range(tail_start, frame_index)):
            fraction = _smoothstep((offset + 1) / blend_frames)
            position[sample_index] = (1.0 - fraction) * position[before] + fraction * position[after]
            opening[sample_index] = (1.0 - fraction) * opening[before] + fraction * opening[after]
            quaternion[sample_index] = orientation([fraction]).as_quat()[0]
    return replace(track, position_m=position, opening_distance_m=opening, quaternion_xyzw=quaternion)


def _smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)
