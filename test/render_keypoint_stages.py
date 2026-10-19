"""Draw filled hand keypoints with the module 6 and module 7 Dex1 grippers.

This script only reads a completed run. It writes two side-by-side MP4 files.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from contracts import GripperPoses, RepairedHands, load_gripper_poses, load_rectified_stereo, load_repaired_hands
from dex1_gripper_drawer import Dex1GripperDrawer


# =============================================================================
# CONFIG: edit these values before running this script.
# =============================================================================
MY_RETARGETING_ROOT = Path(__file__).resolve().parents[1]
# None uses output/latest_run.txt written by run_humanplus_wilor.py.
RUN_DIRECTORY: Path | None = None
OUTPUT_SCALE = 0.5
OUTPUT_FPS = None
VIDEO_CODEC = "mp4v"
# Each worker loads the Dex1 mesh and two video decoders. Using every core
# exhausted memory. None means four processes.
WORKER_COUNT: int | None = None
_DEFAULT_WORKER_COUNT = 4
_WORKER: dict | None = None

_LEFT_COLOR = (0, 165, 255)
_RIGHT_COLOR = (255, 144, 30)
_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 17),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
)


def main() -> None:
    """Write module 6 and module 7 videos of filled keypoints plus the Dex1 gripper."""

    cv2 = _require_cv2()
    run_directory = _resolve_run_directory()
    print(f"读取运行目录: {run_directory}", flush=True)
    capture = load_rectified_stereo(run_directory)
    filled = load_repaired_hands(run_directory / "filled_hands.npz")
    raw_pose = load_gripper_poses(run_directory / "gripper_pose_raw.npz")
    filtered_pose = load_gripper_poses(run_directory / "gripper_poses.npz")
    frame_count = len(capture.frame_indices)
    worker_count = _worker_count(frame_count)
    videos = (
        ("模块6原始夹爪", raw_pose, run_directory / "gripper_pose_module6.mp4"),
        ("模块7滤波夹爪", filtered_pose, run_directory / "gripper_pose_module7.mp4"),
    )
    for label, poses, output_path in videos:
        print(f"开始渲染 {label}：{frame_count} 帧，{worker_count} 个进程 -> {output_path}", flush=True)
        _write_final_video(cv2, capture, filled, poses, output_path, worker_count, label)
        print(f"完成: {output_path}", flush=True)


def _write_final_video(
    cv2, capture, filled: RepairedHands, poses: GripperPoses, output_path: Path, worker_count: int, label: str
) -> None:
    """Render frames in worker processes and write one MP4 in order."""

    right_from_left = np.linalg.inv(np.asarray(capture.right_pinhole.T_rig_from_camera, dtype=float)) @ np.asarray(
        capture.left_pinhole.T_rig_from_camera, dtype=float
    )
    source_fps = _video_fps(cv2, capture.left_rectified_video)
    output_fps = OUTPUT_FPS if OUTPUT_FPS is not None else source_fps
    if output_fps <= 0.0:
        raise ValueError("The rectified video has no positive FPS; set OUTPUT_FPS.")
    frame_count = len(capture.frame_indices)
    frame_dir = output_path.parent / f".{output_path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    shared = _frame_job(capture, filled, poses, right_from_left, 0, frame_count, label)
    shared["frame_dir"] = str(frame_dir)
    ranges = _frame_ranges(frame_count, worker_count)
    writer = None
    written = 0
    try:
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker, initargs=(shared,)) as pool:
            futures = [pool.submit(_render_frame_range, start, stop) for start, stop in ranges]
            while written < frame_count:
                frame_path = frame_dir / f"{written:06d}.jpg"
                if not frame_path.is_file():
                    failed = next((future for future in futures if future.done() and future.exception() is not None), None)
                    if failed is not None:
                        raise failed.exception()
                    time.sleep(0.05)
                    continue
                image = cv2.imread(str(frame_path))
                frame_path.unlink()
                if image is None:
                    raise RuntimeError(f"Cannot read rendered frame {written}.")
                if writer is None:
                    height_px, width_px = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*VIDEO_CODEC), output_fps, (width_px, height_px)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot open overlay writer: {output_path}")
                writer.write(image)
                written += 1
                print(f"{label} 渲染进度 {written}/{frame_count} ({100.0 * written / frame_count:.1f}%)", flush=True)
            for future in futures:
                future.result()
    finally:
        if writer is not None:
            writer.release()
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)


def _init_worker(shared: dict) -> None:
    """Load the gripper mesh once per process and keep the pose arrays."""

    cv2 = _require_cv2()
    cv2.setNumThreads(1)
    global _WORKER
    _WORKER = {**shared, "cv2": cv2, "drawer": Dex1GripperDrawer()}


def _render_frame_range(start: int, stop: int) -> None:
    """Draw a contiguous frame range and write one JPEG per frame."""

    job = _WORKER
    if job is None:
        raise RuntimeError("Render worker was not initialized.")
    cv2 = job["cv2"]
    drawer = job["drawer"]
    frame_dir = Path(job["frame_dir"])
    left_decoder = _open_video(cv2, Path(job["left_video"]))
    right_decoder = _open_video(cv2, Path(job["right_video"]))
    try:
        _seek_video(left_decoder, start)
        _seek_video(right_decoder, start)
        for frame_index in range(start, stop):
            left_ok, left_frame = left_decoder.read()
            right_ok, right_frame = right_decoder.read()
            if not left_ok or left_frame is None or not right_ok or right_frame is None:
                raise RuntimeError(f"Rectified video ended before frame {frame_index}.")
            _draw_view(cv2, drawer, left_frame, job, frame_index, "left")
            _draw_view(cv2, drawer, right_frame, job, frame_index, "right")
            label = job["label"]
            cv2.putText(left_frame, f"{label} 左  frame={frame_index}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(right_frame, f"{label} 右  frame={frame_index}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            image = _resize(cv2, np.hstack((left_frame, right_frame)), OUTPUT_SCALE)
            partial_path = frame_dir / f"{frame_index:06d}.part.jpg"
            final_path = frame_dir / f"{frame_index:06d}.jpg"
            if not cv2.imwrite(str(partial_path), image):
                raise RuntimeError(f"Cannot write rendered frame {frame_index}.")
            partial_path.replace(final_path)
    finally:
        left_decoder.release()
        right_decoder.release()


def _draw_view(cv2, drawer: Dex1GripperDrawer, frame, job: dict, frame_index: int, side: str) -> None:
    from scipy.spatial.transform import Rotation

    camera_from_left = np.asarray(job[f"{side}_from_left"], dtype=float)
    intrinsics = np.asarray(job[f"{side}_intrinsics"], dtype=float)
    image_size = job[f"{side}_image_size"]
    for hand, color in (("left", _LEFT_COLOR), ("right", _RIGHT_COLOR)):
        drawer.draw(
            cv2,
            frame,
            job[f"{hand}_position"][frame_index],
            Rotation.from_quat(job[f"{hand}_quaternion"][frame_index]).as_matrix(),
            float(job[f"{hand}_opening_distance"][frame_index]),
            float(job[f"{hand}_opening_width"][frame_index]),
            camera_from_left,
            intrinsics,
        )
        _draw_projected_hand(cv2, frame, job[f"{hand}_points"][frame_index], color, camera_from_left, intrinsics, image_size)


def _draw_projected_hand(cv2, frame, points_m: np.ndarray, color: tuple[int, int, int], camera_from_left, intrinsics, image_size) -> None:
    width_px, height_px = image_size
    pixels = np.full((21, 2), np.nan)
    for index, point in enumerate(points_m):
        transformed = camera_from_left @ np.array([point[0], point[1], point[2], 1.0])
        if transformed[2] <= 0.0:
            continue
        projected = intrinsics @ transformed[:3]
        uv = projected[:2] / projected[2]
        if 0.0 <= uv[0] < width_px and 0.0 <= uv[1] < height_px:
            pixels[index] = uv
    _draw_pixels(cv2, frame, pixels, color)


def _frame_job(capture, filled: RepairedHands, poses: GripperPoses, right_from_left: np.ndarray, start: int, stop: int, label: str) -> dict:
    return {
        "start": start,
        "stop": stop,
        "label": label,
        "left_video": str(capture.left_rectified_video),
        "right_video": str(capture.right_rectified_video),
        "left_from_left": np.eye(4),
        "right_from_left": right_from_left,
        "left_intrinsics": np.asarray(capture.left_pinhole.intrinsics, dtype=float),
        "right_intrinsics": np.asarray(capture.right_pinhole.intrinsics, dtype=float),
        "left_image_size": capture.left_pinhole.image_size_px,
        "right_image_size": capture.right_pinhole.image_size_px,
        "left_points": np.asarray(filled.left.points_m, dtype=float),
        "right_points": np.asarray(filled.right.points_m, dtype=float),
        "left_position": np.asarray(poses.left.position_m, dtype=float),
        "right_position": np.asarray(poses.right.position_m, dtype=float),
        "left_quaternion": np.asarray(poses.left.quaternion_xyzw, dtype=float),
        "right_quaternion": np.asarray(poses.right.quaternion_xyzw, dtype=float),
        "left_opening_distance": np.asarray(poses.left.opening_distance_m, dtype=float),
        "right_opening_distance": np.asarray(poses.right.opening_distance_m, dtype=float),
        "left_opening_width": np.asarray(poses.left.opening_width, dtype=float),
        "right_opening_width": np.asarray(poses.right.opening_width, dtype=float),
    }


def _frame_ranges(frame_count: int, worker_count: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for worker_index in range(worker_count):
        stop = frame_count * (worker_index + 1) // worker_count
        if stop > start:
            ranges.append((start, stop))
        start = stop
    return ranges


def _worker_count(frame_count: int) -> int:
    requested = _DEFAULT_WORKER_COUNT if WORKER_COUNT is None else WORKER_COUNT
    if requested < 1:
        raise ValueError("WORKER_COUNT must be positive.")
    return max(1, min(requested, frame_count))


def _video_fps(cv2, video_path: Path) -> float:
    capture = _open_video(cv2, video_path)
    try:
        return float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        capture.release()


def _seek_video(capture, frame_index: int) -> None:
    if frame_index > 0:
        capture.set(1, frame_index)  # cv2.CAP_PROP_POS_FRAMES


def _draw_pixels(cv2, frame, pixels: np.ndarray, color: tuple[int, int, int]) -> None:
    for start, end in _CONNECTIONS:
        if np.isfinite(pixels[start]).all() and np.isfinite(pixels[end]).all():
            cv2.line(frame, _pixel(pixels[start]), _pixel(pixels[end]), color, 2, cv2.LINE_AA)
    for point in pixels:
        if np.isfinite(point).all():
            cv2.circle(frame, _pixel(point), 3, color, -1, cv2.LINE_AA)


def _pixel(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


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
        raise RuntimeError("OpenCV is required to render keypoint videos.") from error
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
