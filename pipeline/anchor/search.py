"""Per-arm top-5 screening and the 25-pair check from Ego2Robot appendix A.4.

Left and right bases are generated separately. Each side keeps five placements.
Every left-right pair is scored again, and the pair with the higher mean score
is kept. A side with no valid frames, or with every candidate pruned, stays
absent. This module does not implement IK.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anchor.candidates import AnchorSearchConfig, base_positions, base_rotations
from anchor.keyframes import select_keyframes
from anchor.prune import is_base_admissible
from anchor.score import AnchorScore, score_placement
from arm_ik.model import ArmModel
from arm_ik.solver import ArmSolver
from contracts import ArmAnchor, ArmAnchors, GripperPoseTrack, GripperPoses
from se3 import quaternion_xyzw, transform


@dataclass(frozen=True)
class _Placement:
    position_m: np.ndarray
    rotation: np.ndarray
    score: AnchorScore


def search_arm_anchors(
    poses: GripperPoses,
    solvers: dict[str, ArmSolver],
    models: dict[str, ArmModel],
    config: AnchorSearchConfig | None = None,
) -> ArmAnchors:
    """Search two independent arm bases for one gripper trajectory.

    Args:
        poses: Module 7 gripper poses in the left rectified camera.
        solvers: One feasibility solver per side that still has valid frames.
        models: Arm models that supply reach. Shoulder width is not used.
        config: Paper grid and score constants. ``None`` uses those constants.
    Returns:
        Left and right anchors. An absent side has ``present`` false.
    """

    settings = AnchorSearchConfig() if config is None else config
    ranked = {
        side: _rank_side(getattr(poses, side), solvers.get(side), models[side], settings)
        for side in ("left", "right")
    }
    if ranked["left"] and ranked["right"]:
        return _best_pair(poses, solvers, models, ranked["left"][: settings.top_k], ranked["right"][: settings.top_k], settings)
    left = _anchor_from_top(poses.left, ranked["left"])
    right = _anchor_from_top(poses.right, ranked["right"])
    return ArmAnchors(left=left, right=right, pair_score=_single_score(left, right))


def _rank_side(
    track: GripperPoseTrack,
    solver: ArmSolver | None,
    model: ArmModel,
    config: AnchorSearchConfig,
) -> list[_Placement]:
    valid = ~np.asarray(track.invalid_mask, dtype=bool)
    if not np.any(valid):
        return []
    if solver is None:
        raise ValueError(f"Arm '{track.side}' has valid frames but no IK solver was provided.")
    points = np.asarray(track.position_m, dtype=float)
    quaternions = np.asarray(track.quaternion_xyzw, dtype=float)
    keyframes = select_keyframes(points, quaternions, valid, config.maximum_keyframes)
    positions = base_positions(track.side, model.reach_m, config)
    admissible = [
        position
        for position in positions
        if is_base_admissible(position, points[valid], model.reach_m, config)
    ]
    ranked: list[_Placement] = []
    for position in admissible:
        for rotation in base_rotations(config):
            base_pose = transform(rotation, position)
            score = score_placement(solver, base_pose, points, quaternions, keyframes, model.reach_m, config)
            ranked.append(_Placement(position_m=position, rotation=rotation, score=score))
    ranked.sort(key=lambda placement: placement.score.score, reverse=True)
    return ranked


def _best_pair(
    poses: GripperPoses,
    solvers: dict[str, ArmSolver],
    models: dict[str, ArmModel],
    left_candidates: list[_Placement],
    right_candidates: list[_Placement],
    config: AnchorSearchConfig,
) -> ArmAnchors:
    best_score = None
    best_pair: tuple[_Placement, _Placement, AnchorScore, AnchorScore] | None = None
    for left in left_candidates:
        for right in right_candidates:
            left_score = _rescore(poses.left, solvers["left"], models["left"], left, config)
            right_score = _rescore(poses.right, solvers["right"], models["right"], right, config)
            pair_score = 0.5 * (left_score.score + right_score.score)
            if best_score is None or pair_score > best_score:
                best_score = pair_score
                best_pair = (left, right, left_score, right_score)
    if best_pair is None or best_score is None:
        raise RuntimeError("Pair verification ran without candidates.")
    left, right, left_score, right_score = best_pair
    return ArmAnchors(
        left=_present_anchor("left", left, left_score),
        right=_present_anchor("right", right, right_score),
        pair_score=float(best_score),
    )


def _rescore(
    track: GripperPoseTrack,
    solver: ArmSolver,
    model: ArmModel,
    placement: _Placement,
    config: AnchorSearchConfig,
) -> AnchorScore:
    valid = ~np.asarray(track.invalid_mask, dtype=bool)
    points = np.asarray(track.position_m, dtype=float)
    quaternions = np.asarray(track.quaternion_xyzw, dtype=float)
    keyframes = select_keyframes(points, quaternions, valid, config.maximum_keyframes)
    return score_placement(
        solver,
        transform(placement.rotation, placement.position_m),
        points,
        quaternions,
        keyframes,
        model.reach_m,
        config,
    )


def _anchor_from_top(track: GripperPoseTrack, ranked: list[_Placement]) -> ArmAnchor:
    if not ranked:
        return _absent_anchor(track.side)
    winner = ranked[0]
    return _present_anchor(track.side, winner, winner.score)


def _present_anchor(side: str, placement: _Placement, score: AnchorScore) -> ArmAnchor:
    return ArmAnchor(
        side=side,
        present=True,
        position_m=np.asarray(placement.position_m, dtype=float),
        quaternion_xyzw=quaternion_xyzw(placement.rotation),
        score=float(score.score),
        feasibility_rate=float(score.feasibility_rate),
        mean_reach_fraction=float(score.mean_reach_fraction),
    )


def _absent_anchor(side: str) -> ArmAnchor:
    return ArmAnchor(
        side=side,
        present=False,
        position_m=np.full(3, np.nan),
        quaternion_xyzw=np.full(4, np.nan),
        score=float("nan"),
        feasibility_rate=float("nan"),
        mean_reach_fraction=float("nan"),
    )


def _single_score(left: ArmAnchor, right: ArmAnchor) -> float:
    if left.present and right.present:
        return 0.5 * (left.score + right.score)
    if left.present:
        return float(left.score)
    if right.present:
        return float(right.score)
    return float("nan")
