"""Ego2Robot appendix A.4 candidate grid, in the left rectified camera.

Positions are offsets from the camera origin, scaled by the arm reach. Lateral
is camera X and is sign-flipped per arm. Forward is camera Z. Vertical is
camera Y. This module does not prune, score, or solve IK.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AnchorSearchConfig:
    """Paper constants for the base grid, the prune, and the score.

    Fractions multiply the arm reach in meters. Orientation angles are degrees.
    """

    lateral_fractions: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
    forward_fractions: tuple[float, ...] = (-0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
    vertical_fractions: tuple[float, ...] = (0.4, 0.2, 0.0, -0.2, -0.4)
    pitch_deg: tuple[float, ...] = (30.0, 45.0, 60.0)
    yaw_deg: tuple[float, ...] = (-45.0, -20.0, 0.0, 20.0, 45.0)
    roll_deg: tuple[float, ...] = (-15.0, 0.0, 15.0)
    minimum_base_distance_m: float = 0.20
    minimum_reach_m: float = 0.08
    maximum_stretch_fraction: float = 0.9
    target_reach_fraction: float = 0.65
    reach_penalty: float = 5.0
    maximum_keyframes: int = 20
    top_k: int = 5

    def __post_init__(self) -> None:
        if self.minimum_base_distance_m <= 0.0 or self.minimum_reach_m <= 0.0:
            raise ValueError("Distance gates must be positive.")
        if not 0.0 < self.maximum_stretch_fraction:
            raise ValueError("maximum_stretch_fraction must be positive.")
        if self.maximum_keyframes < 1 or self.top_k < 1:
            raise ValueError("Keyframe and top-k counts must be positive.")


def base_positions(side: str, reach_m: float, config: AnchorSearchConfig) -> np.ndarray:
    """Return camera-frame base translations, shape ``(n, 3)``, meters.

    The grid is anchored at the camera origin, as in Ego2Robot appendix A.4.
    It does not move with the hand trajectory.

    Args:
        side: ``left`` uses a negative lateral offset. ``right`` uses positive.
        reach_m: Morphology reach ``r`` from the arm model.
        config: Paper grid fractions.
    """

    if reach_m <= 0.0:
        raise ValueError("reach_m must be positive.")
    sign = _lateral_sign(side)
    positions = [
        np.array([sign * lateral * reach_m, vertical * reach_m, forward * reach_m], dtype=float)
        for lateral in config.lateral_fractions
        for forward in config.forward_fractions
        for vertical in config.vertical_fractions
    ]
    return np.stack(positions)


def base_rotations(config: AnchorSearchConfig) -> list[np.ndarray]:
    """Yaw-pitch-roll rotations of the arm base in the camera frame.

    The composition is intrinsic ZYX: yaw about camera Z, then pitch about the
    new Y, then roll about the new X.
    """

    rotations = []
    for pitch in config.pitch_deg:
        for yaw in config.yaw_deg:
            for roll in config.roll_deg:
                rotations.append(Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix())
    return rotations


def _lateral_sign(side: str) -> float:
    if side == "right":
        return 1.0
    if side == "left":
        return -1.0
    raise ValueError("Arm side must be 'left' or 'right'.")
