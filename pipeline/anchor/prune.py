"""Geometric rejection from Ego2Robot appendix A.4.

A base is removed when it sits too close to the camera, or when any valid
gripper point is nearer than 0.08 m or farther than 0.9 reach. Orientation is
not an input. This module does not call IK.
"""

from __future__ import annotations

import numpy as np

from anchor.candidates import AnchorSearchConfig


def is_base_admissible(
    base_position_m: np.ndarray,
    trajectory_m: np.ndarray,
    reach_m: float,
    config: AnchorSearchConfig,
) -> bool:
    """Return whether one base translation survives the paper prune.

    Args:
        base_position_m: Base origin in the left rectified camera, meters.
        trajectory_m: Valid gripper positions, shape ``(n, 3)``, meters.
        reach_m: Arm reach ``r``.
        config: Distance gates from the paper.
    """

    if reach_m <= 0.0:
        raise ValueError("reach_m must be positive.")
    points = np.asarray(trajectory_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return False
    if float(np.linalg.norm(base_position_m)) < config.minimum_base_distance_m:
        return False
    distances = np.linalg.norm(points - np.asarray(base_position_m, dtype=float), axis=1)
    if float(distances.min()) < config.minimum_reach_m:
        return False
    if float(distances.max()) > config.maximum_stretch_fraction * reach_m:
        return False
    return True
