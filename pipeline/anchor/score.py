"""Ego2Robot base score, equation 8.

``S = FR - 5 |mean_reach_fraction - 0.65|``. ``FR`` is the IK feasibility rate
on the selected keyframes. The reach fraction uses keyframe distance from the
base divided by the arm reach. This module calls the injected solver and does
not generate candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anchor.candidates import AnchorSearchConfig
from arm_ik.solver import ArmSolver
from se3 import invert_transform, pose_from_translation_quaternion


@dataclass(frozen=True)
class AnchorScore:
    """One base placement's paper score and the two terms that produce it."""

    score: float
    feasibility_rate: float
    mean_reach_fraction: float


def score_placement(
    solver: ArmSolver,
    base_pose: np.ndarray,
    position_m: np.ndarray,
    quaternion_xyzw: np.ndarray,
    keyframes: np.ndarray,
    reach_m: float,
    config: AnchorSearchConfig,
) -> AnchorScore:
    """Score one base by solving IK at each keyframe.

    Args:
        solver: Single-frame feasibility solver for this arm.
        base_pose: 4x4 camera-from-base transform. Translation is meters.
        position_m: Gripper positions in the camera, shape ``(n, 3)``.
        quaternion_xyzw: Gripper orientations, scipy xyzw, shape ``(n, 4)``.
        keyframes: Frame indices. The paper uses at most 20.
        reach_m: Arm reach ``r``.
        config: Holds the 0.65 target fraction and the penalty weight.
    Returns:
        ``S``, the feasibility rate, and the mean normalized reach.
    """

    if reach_m <= 0.0:
        raise ValueError("reach_m must be positive.")
    frames = np.asarray(keyframes, dtype=int)
    if len(frames) == 0:
        raise ValueError("score_placement requires at least one keyframe.")
    base_from_camera = invert_transform(base_pose)
    successes = []
    distances = []
    for frame_index in frames:
        camera_from_jaw = pose_from_translation_quaternion(position_m[frame_index], quaternion_xyzw[frame_index])
        target_in_base = base_from_camera @ camera_from_jaw
        result = solver.solve(target_in_base)
        successes.append(result.succeeded)
        distances.append(float(np.linalg.norm(target_in_base[:3, 3])))
    feasibility_rate = float(np.mean(successes))
    mean_reach_fraction = float(np.mean(distances) / reach_m)
    score = feasibility_rate - config.reach_penalty * abs(mean_reach_fraction - config.target_reach_fraction)
    return AnchorScore(
        score=score,
        feasibility_rate=feasibility_rate,
        mean_reach_fraction=mean_reach_fraction,
    )
