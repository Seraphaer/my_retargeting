"""Module 5: fill triangulated hands with the SSH whole-frame keypoint rules.

The order is ``gate_frames``, zero rejected frames, ``detect_invalid_frames``,
then ``fill_keypoints_gap_aware``. Palm smoothing and gripper retarget are not
applied. Each hand is independent. The triangulation array is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from contracts import HandTrack3D, RepairedHandTrack, RepairedHands, TriangulatedHands


WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
OUTLIER_CHECK_POINTS = [WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
_SPAN_RANGE_M = (0.055, 0.135)
_Z_RANGE_M = (0.10, 1.5)
_MAX_PAIRWISE_M = 0.25


@dataclass(frozen=True)
class FillConfig:
    """SSH thresholds for the post-triangulation 21-point fill.

    Args:
        frames_per_second: Video rate used to turn an adjacent-frame
            displacement into meters per second. SSH multiplies by this rate
            instead of using timestamp deltas.
        maximum_speed_mps: Whole-frame speed limit. SSH uses ``2.5``.
        long_gap_frames: Longer gaps hold the previous pose. SSH uses ``10``.
    """

    frames_per_second: float
    maximum_speed_mps: float = 2.5
    long_gap_frames: int = 10

    def __post_init__(self) -> None:
        if self.frames_per_second <= 0.0 or self.maximum_speed_mps <= 0.0:
            raise ValueError("Frame rate and maximum speed must be positive.")
        if self.long_gap_frames < 0:
            raise ValueError("long_gap_frames must be non-negative.")


def fill_triangulated_hands(tracks: TriangulatedHands, config: FillConfig) -> RepairedHands:
    """Gate and fill both triangulated hands.

    Args:
        tracks: SSH-style measurements in the left rectified camera. Invalid
            points are already zeros.
        config: Speed, gap length, and the video frame rate.
    Returns:
        Filled poses plus the frame-level invalid mask that produced them.
    """

    return RepairedHands(
        left=_fill_hand(tracks.left, config),
        right=_fill_hand(tracks.right, config),
    )


def _fill_hand(raw: HandTrack3D, config: FillConfig) -> RepairedHandTrack:
    points = np.array(raw.points_m, dtype=np.float64, copy=True)
    accepted = _gate_frames(points, np.asarray(raw.point_valid_mask, dtype=bool))
    points[~accepted] = 0.0
    points = np.nan_to_num(points)
    invalid = _detect_invalid_frames(points, config.frames_per_second, config.maximum_speed_mps)
    filled = _fill_keypoints_gap_aware(points.reshape(points.shape[0], -1), invalid, config.long_gap_frames)
    return RepairedHandTrack(raw=raw, points_m=filled.reshape(points.shape), invalid_mask=invalid)


def _gate_frames(points_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Copy ssh_run_all ``gino_rig.gate_frames``."""

    accepted = valid_mask.all(axis=1)
    span = np.linalg.norm(points_m[:, MIDDLE_MCP] - points_m[:, WRIST], axis=1)
    depth = points_m[..., 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        finite_depth = np.where(accepted[:, None], depth, np.nan)
        depth_min = np.where(accepted, np.nanmin(finite_depth, axis=1), np.inf)
        depth_max = np.where(accepted, np.nanmax(finite_depth, axis=1), -np.inf)
    max_pairwise = np.full(len(points_m), np.inf)
    accepted_index = np.flatnonzero(accepted)
    if len(accepted_index):
        selected = points_m[accepted_index]
        differences = selected[:, :, None, :] - selected[:, None, :, :]
        max_pairwise[accepted_index] = np.linalg.norm(differences, axis=-1).reshape(len(accepted_index), -1).max(axis=1)
    return (
        accepted
        & (span > _SPAN_RANGE_M[0])
        & (span < _SPAN_RANGE_M[1])
        & (depth_min > _Z_RANGE_M[0])
        & (depth_max < _Z_RANGE_M[1])
        & (max_pairwise < _MAX_PAIRWISE_M)
    )


def _detect_invalid_frames(points_m: np.ndarray, fps: float, maximum_speed_mps: float) -> np.ndarray:
    """Copy ssh_run_all ``detect_invalid_frames``."""

    missing = np.array([np.allclose(points_m[frame], 0.0, atol=1e-6) for frame in range(len(points_m))])
    checked = points_m[:, OUTLIER_CHECK_POINTS, :]
    displacement = np.zeros((len(points_m), len(OUTLIER_CHECK_POINTS)))
    displacement[1:] = np.linalg.norm(checked[1:] - checked[:-1], axis=-1) * fps
    straddles_missing = missing[1:] | missing[:-1]
    displacement[1:][straddles_missing] = 0.0
    return missing | (displacement.max(axis=1) > maximum_speed_mps)


def _fill_keypoints_gap_aware(points_flat: np.ndarray, invalid: np.ndarray, long_gap_frames: int) -> np.ndarray:
    """Copy ssh_run_all ``fill_keypoints_gap_aware``."""

    if (~invalid).sum() == 0:
        return points_flat.copy()
    filled = _interpolate_channels(points_flat, invalid)
    for start, end, length in _gap_segments(invalid):
        if length <= long_gap_frames or start == 0:
            continue
        filled[start:end] = filled[start - 1]
    return filled


def _interpolate_channels(values: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Copy ssh_run_all ``interpolate_channels``."""

    valid = ~invalid
    index = np.arange(len(values))
    if valid.sum() == 0:
        return values.copy()
    filled = np.empty_like(values, dtype=float)
    for channel in range(values.shape[1]):
        filled[:, channel] = np.interp(index, index[valid], values[valid, channel])
    return filled


def _gap_segments(invalid: np.ndarray) -> list[tuple[int, int, int]]:
    """Copy ssh_run_all ``gap_segments``. End is exclusive."""

    segments: list[tuple[int, int, int]] = []
    frame_index = 0
    while frame_index < len(invalid):
        if invalid[frame_index]:
            start = frame_index
            while frame_index < len(invalid) and invalid[frame_index]:
                frame_index += 1
            segments.append((start, frame_index, frame_index - start))
        else:
            frame_index += 1
    return segments
