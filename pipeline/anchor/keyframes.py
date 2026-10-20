"""Up to 20 keyframes covering the spatial extremes of one gripper trajectory.

Frames are chosen from the valid mask only. The set mixes the largest
displacements from the valid-position centroid with the largest adjacent
orientation steps. This module does not score or solve.
"""

from __future__ import annotations

import numpy as np


def select_keyframes(
    position_m: np.ndarray,
    quaternion_xyzw: np.ndarray,
    valid_mask: np.ndarray,
    maximum_keyframes: int,
) -> np.ndarray:
    """Return sorted frame indices, at most ``maximum_keyframes``.

    Args:
        position_m: Gripper positions in meters, shape ``(n, 3)``.
        quaternion_xyzw: Gripper orientations, scipy xyzw, shape ``(n, 4)``.
        valid_mask: True where the frame may enter the search.
        maximum_keyframes: Paper cap, normally 20.
    Returns:
        Increasing frame indices. Empty when no frame is valid.
    """

    if maximum_keyframes < 1:
        raise ValueError("maximum_keyframes must be positive.")
    valid = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if len(valid) <= maximum_keyframes:
        return valid.astype(np.int64)
    points = np.asarray(position_m, dtype=float)[valid]
    displacement = np.linalg.norm(points - points.mean(axis=0), axis=1)
    orientation_step = _adjacent_angles(np.asarray(quaternion_xyzw, dtype=float)[valid])
    chosen: list[int] = []
    for displacement_rank, angle_rank in zip(np.argsort(-displacement), np.argsort(-orientation_step)):
        for local_index in (int(displacement_rank), int(angle_rank)):
            frame_index = int(valid[local_index])
            if frame_index not in chosen:
                chosen.append(frame_index)
            if len(chosen) == maximum_keyframes:
                return np.array(sorted(chosen), dtype=np.int64)
    return np.array(sorted(chosen), dtype=np.int64)


def _adjacent_angles(quaternion_xyzw: np.ndarray) -> np.ndarray:
    angles = np.zeros(len(quaternion_xyzw), dtype=float)
    for index in range(1, len(quaternion_xyzw)):
        dot = abs(float(np.dot(quaternion_xyzw[index], quaternion_xyzw[index - 1])))
        angles[index] = 2.0 * np.arccos(float(np.clip(dot, -1.0, 1.0)))
    return angles
