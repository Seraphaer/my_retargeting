"""Re-search arm bases from a saved gripper trajectory, then render the overlay.

WiLoR, triangulation, and gripper mapping are not run again. Those files stay
in the run directory. This script overwrites ``arm_anchors.npz`` and
``arm_ik_overlay.mp4`` only.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
TEST_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from arm_ik.model import ArmModel
from contracts import load_gripper_poses, save_arm_anchors
import render_arm_ik


# =============================================================================
# CONFIG: edit these values before running this script.
# =============================================================================
MY_RETARGETING_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = MY_RETARGETING_ROOT / "output" / "humanplus_wilor_20261020_124844"
URDF_PATH = MY_RETARGETING_ROOT / "urdf" / "g1_29dof_with_dex1_1.urdf"
MAXIMUM_FRAMES: int | None = None


def main() -> None:
    """Search bases for the saved gripper poses and write a new overlay video."""

    pose_path = RUN_DIRECTORY / "gripper_poses.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Saved gripper poses are missing: {pose_path}")
    poses = load_gripper_poses(pose_path)
    print(f"复用夹爪轨迹: {pose_path}", flush=True)
    print(f"  frames: {len(poses.left.frame_indices)}", flush=True)
    solvers = {
        "left": _stage81().make_mink_solver(URDF_PATH, "left"),
        "right": _stage81().make_mink_solver(URDF_PATH, "right"),
    }
    models = {
        "left": ArmModel.from_urdf(URDF_PATH, "left"),
        "right": ArmModel.from_urdf(URDF_PATH, "right"),
    }
    anchors = _stage82().auto_anchor(poses, solvers, models)
    anchor_path = save_arm_anchors(anchors, RUN_DIRECTORY / "arm_anchors.npz")
    print(f"底座已写入: {anchor_path}", flush=True)
    print(f"  left present: {anchors.left.present} score={anchors.left.score:.3f}", flush=True)
    print(f"  right present: {anchors.right.present} score={anchors.right.score:.3f}", flush=True)
    render_arm_ik.RUN_DIRECTORY = RUN_DIRECTORY
    render_arm_ik.URDF_PATH = URDF_PATH
    render_arm_ik.MAXIMUM_FRAMES = MAXIMUM_FRAMES
    render_arm_ik.main()


def _stage81():
    return _load_stage("8.1_solve_arm_ik.py")


def _stage82():
    return _load_stage("8.2_auto_anchor.py")


def _load_stage(filename: str):
    path = PIPELINE_DIR / filename
    spec = spec_from_file_location(filename.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}.")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
