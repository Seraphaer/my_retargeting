"""Overlay solved G1 arms on one rectified stereo run.

Reads the module 7 gripper poses and the bases saved by module 8.2. Every
frame calls ``8.1_solve_arm_ik``. This script does not search a base and does
not write the trajectory back.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from arm_ik.model import ArmModel
from contracts import ArmAnchor, GripperPoseTrack, load_arm_anchors, load_gripper_poses, load_rectified_stereo
from dex1_gripper_drawer import Dex1GripperDrawer
from se3 import invert_transform, pose_from_translation_quaternion


# =============================================================================
# CONFIG: edit these values before running this script.
# =============================================================================
MY_RETARGETING_ROOT = Path(__file__).resolve().parents[1]
# None uses output/latest_run.txt written by run_humanplus_wilor.py.
RUN_DIRECTORY: Path | None = None
URDF_PATH = MY_RETARGETING_ROOT / "urdf" / "g1_29dof_with_dex1_1.urdf"
OUTPUT_NAME = "arm_ik_overlay.mp4"
OUTPUT_SCALE = 0.5
OUTPUT_FPS = None
VIDEO_CODEC = "mp4v"
MAXIMUM_FRAMES: int | None = None
MESH_STRIDE = 12

_LEFT_COLOR = (180, 140, 90)
_RIGHT_COLOR = (90, 140, 180)


def main() -> None:
    """Solve each frame with module 8.1 and write one side-by-side MP4."""

    cv2 = _require_cv2()
    run_directory = _resolve_run_directory()
    print(f"读取运行目录: {run_directory}", flush=True)
    capture = load_rectified_stereo(run_directory)
    poses = load_gripper_poses(run_directory / "gripper_poses.npz")
    anchors = load_arm_anchors(run_directory / "arm_anchors.npz")
    stage = _load_stage(PIPELINE_DIR / "8.1_solve_arm_ik.py")
    arms = {
        "left": _ArmView("left", URDF_PATH, stage, _LEFT_COLOR),
        "right": _ArmView("right", URDF_PATH, stage, _RIGHT_COLOR),
    }
    drawer = Dex1GripperDrawer()
    right_from_left = np.linalg.inv(np.asarray(capture.right_pinhole.T_rig_from_camera, dtype=float)) @ np.asarray(
        capture.left_pinhole.T_rig_from_camera, dtype=float
    )
    frame_count = len(capture.frame_indices) if MAXIMUM_FRAMES is None else min(MAXIMUM_FRAMES, len(capture.frame_indices))
    output_path = run_directory / OUTPUT_NAME
    left_video = _open_video(cv2, capture.left_rectified_video)
    right_video = _open_video(cv2, capture.right_rectified_video)
    writer = None
    try:
        source_fps = float(left_video.get(cv2.CAP_PROP_FPS) or 30.0)
        output_fps = source_fps if OUTPUT_FPS is None else OUTPUT_FPS
        for frame_index in range(frame_count):
            left_ok, left_frame = left_video.read()
            right_ok, right_frame = right_video.read()
            if not left_ok or left_frame is None or not right_ok or right_frame is None:
                raise RuntimeError(f"Rectified video ended before frame {frame_index}.")
            solved = {
                side: arms[side].solve_frame(getattr(poses, side), getattr(anchors, side), frame_index, stage)
                for side in ("left", "right")
            }
            _draw_view(cv2, drawer, left_frame, solved, np.eye(4), capture.left_pinhole.intrinsics, "左", frame_index)
            _draw_view(
                cv2,
                drawer,
                right_frame,
                solved,
                right_from_left,
                capture.right_pinhole.intrinsics,
                "右",
                frame_index,
            )
            image = _resize(cv2, np.hstack((left_frame, right_frame)), OUTPUT_SCALE)
            if writer is None:
                height_px, width_px = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*VIDEO_CODEC), output_fps, (width_px, height_px)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open overlay writer: {output_path}")
            writer.write(image)
            print(f"手臂渲染进度 {frame_index + 1}/{frame_count}", flush=True)
    finally:
        left_video.release()
        right_video.release()
        if writer is not None:
            writer.release()
    print(f"完成: {output_path}", flush=True)


class _ArmView:
    """One arm's meshes plus the module 8.1 solver used to pose them."""

    def __init__(self, side: str, urdf_path: Path, stage, color: tuple[int, int, int]) -> None:
        self.side = side
        self.color = color
        self.model = ArmModel.from_urdf(urdf_path, side)
        self.solver = stage.make_mink_solver(urdf_path, side)
        self.meshes = [_load_mesh(joint.mesh_file, MESH_STRIDE) for joint in self.model.joints]

    def solve_frame(self, track: GripperPoseTrack, anchor: ArmAnchor, frame_index: int, stage) -> dict[str, object]:
        """Solve one frame. A missing base or a failed solve carries no joints."""

        position = np.asarray(track.position_m[frame_index], dtype=float)
        quaternion = np.asarray(track.quaternion_xyzw[frame_index], dtype=float)
        opening_distance = float(track.opening_distance_m[frame_index])
        opening_width = float(track.opening_width[frame_index])
        if bool(track.invalid_mask[frame_index]):
            return _marked("无效", position, quaternion, opening_distance, opening_width)
        if not anchor.present:
            return _marked("无底座", position, quaternion, opening_distance, opening_width)
        camera_from_jaw = pose_from_translation_quaternion(position, quaternion)
        camera_from_base = pose_from_translation_quaternion(anchor.position_m, anchor.quaternion_xyzw)
        result = stage.solve_gripper_in_base(invert_transform(camera_from_base) @ camera_from_jaw, self.solver)
        if not result.succeeded:
            return _marked("IK失败", position, quaternion, opening_distance, opening_width)
        jaw, links = self.model.forward(result.joint_rad)
        camera_jaw = camera_from_base @ jaw
        camera_links = [camera_from_base @ links[joint.child_link] for joint in self.model.joints]
        return {
            "status": "ok",
            "position": camera_jaw[:3, 3],
            "rotation": camera_jaw[:3, :3],
            "opening_distance": opening_distance,
            "opening_width": opening_width,
            "links": camera_links,
            "color": self.color,
            "meshes": self.meshes,
        }


def _marked(status: str, position: np.ndarray, quaternion: np.ndarray, opening_distance: float, opening_width: float) -> dict:
    rotation = np.eye(3)
    if np.all(np.isfinite(quaternion)):
        rotation = Rotation.from_quat(quaternion).as_matrix()
    return {
        "status": status,
        "position": position,
        "rotation": rotation,
        "opening_distance": opening_distance,
        "opening_width": opening_width,
        "links": [],
        "color": (0, 0, 255),
        "meshes": [],
    }


def _draw_view(cv2, drawer: Dex1GripperDrawer, frame, solved: dict[str, dict], camera_from_left, intrinsics, label: str, frame_index: int) -> None:
    for side in ("left", "right"):
        item = solved[side]
        if item["status"] == "ok":
            for mesh, link_pose in zip(item["meshes"], item["links"]):
                _draw_mesh(cv2, frame, mesh, link_pose, camera_from_left, intrinsics, item["color"])
            drawer.draw(
                cv2,
                frame,
                item["position"],
                item["rotation"],
                item["opening_distance"],
                item["opening_width"],
                camera_from_left,
                intrinsics,
            )
        else:
            _draw_marker(cv2, frame, item["position"], camera_from_left, intrinsics, f"{side} {item['status']}")
    cv2.putText(frame, f"8.1 手臂 {label}  frame={frame_index}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_marker(cv2, frame, position_m, camera_from_left, intrinsics, text: str) -> None:
    pixel = _project_point(position_m, camera_from_left, intrinsics)
    if pixel is None:
        return
    cv2.drawMarker(frame, pixel, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (pixel[0] + 8, pixel[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def _draw_mesh(cv2, frame, mesh, link_pose, camera_from_left, intrinsics, color: tuple[int, int, int]) -> None:
    vertices = mesh["vertices"] @ np.asarray(link_pose)[:3, :3].T + np.asarray(link_pose)[:3, 3]
    camera = (camera_from_left[:3, :3] @ vertices.T).T + camera_from_left[:3, 3]
    depth = camera[:, 2]
    projected = (np.asarray(intrinsics, dtype=float) @ camera.T).T
    pixels = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
    height_px, width_px = frame.shape[:2]
    for face in mesh["faces"]:
        if np.any(depth[face] <= 1e-3):
            continue
        face_pixels = np.round(pixels[face]).astype(np.int32)
        if (
            face_pixels[:, 0].max() < 0
            or face_pixels[:, 1].max() < 0
            or face_pixels[:, 0].min() >= width_px
            or face_pixels[:, 1].min() >= height_px
        ):
            continue
        cv2.fillConvexPoly(frame, face_pixels, color, cv2.LINE_AA)


def _project_point(position_m, camera_from_left, intrinsics) -> tuple[int, int] | None:
    if not np.all(np.isfinite(position_m)):
        return None
    camera = camera_from_left @ np.array([position_m[0], position_m[1], position_m[2], 1.0])
    if camera[2] <= 1e-3:
        return None
    projected = np.asarray(intrinsics, dtype=float) @ camera[:3]
    return int(round(projected[0] / projected[2])), int(round(projected[1] / projected[2]))


def _load_mesh(path: str, stride: int) -> dict[str, np.ndarray]:
    payload = Path(path).read_bytes()
    triangle_count = int.from_bytes(payload[80:84], "little")
    records = np.frombuffer(payload, dtype=np.uint8, offset=84, count=triangle_count * 50).reshape(triangle_count, 50)
    vertices = np.ascontiguousarray(records[:, 12:48]).view("<f4").reshape(triangle_count * 3, 3).astype(np.float64)
    faces = np.arange(0, len(vertices) - 2, 3 * stride).reshape(-1, 1) + np.array([0, 1, 2])
    faces = faces[faces[:, 2] < len(vertices)]
    return {"vertices": vertices, "faces": faces}


def _load_stage(path: Path):
    spec = spec_from_file_location(path.stem.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}.")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resize(cv2, image, scale: float):
    if scale == 1.0:
        return image
    height_px, width_px = image.shape[:2]
    size = (max(1, int(round(width_px * scale))), max(1, int(round(height_px * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _open_video(cv2, path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open rectified video: {path}")
    return capture


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to render the arm video.") from error
    return cv2


def _resolve_run_directory() -> Path:
    if RUN_DIRECTORY is not None:
        return RUN_DIRECTORY
    pointer = MY_RETARGETING_ROOT / "output" / "latest_run.txt"
    if not pointer.is_file():
        raise FileNotFoundError("Set RUN_DIRECTORY, or run run_humanplus_wilor.py first so latest_run.txt exists.")
    return Path(pointer.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
