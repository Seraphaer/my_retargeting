"""Draw the Dex1-1 parallel gripper from its URDF visual meshes.

The jaw pose uses columns ``[closing, up, approach]``. Dex1 fingers slide on
local X and point along local Y, so the mesh frame is ``[closing, approach, -up]``.
The base is pulled back along the approach axis until the fingertip links sit
on the pinch point. Opening follows the measured thumb-index distance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parents[1] / "urdf" / "dex1_1" / "meshes"
_CLOSED_Q = 0.0245
_OPEN_Q = -0.02
_ZERO_GAP_M = 0.0811
_BASE_STRIDE = 24
_FINGER_ROOT_STRIDE = 3

_LINKS = (
    ("base_link.STL", (0.79216, 0.81961, 0.93333), "base"),
    ("Link1_1.STL", (0.79216, 0.81961, 0.93333), "finger1"),
    ("Link1_2.STL", (0.89804, 0.91765, 0.92941), "finger1_2"),
    ("Link1_3.STL", (0.29804, 0.29804, 0.29804), "finger1_3"),
    ("Link2_1.STL", (0.79216, 0.81961, 0.93333), "finger2"),
    ("Link2_2.STL", (0.89804, 0.91765, 0.92941), "finger2_2"),
    ("Link2_3.STL", (0.29804, 0.29804, 0.29804), "finger2_3"),
)


class Dex1GripperDrawer:
    """Project one Dex1 gripper into a camera image."""

    def __init__(self) -> None:
        self._parts = [_load_part(name, color, role) for name, color, role in _LINKS]
        self._standoff_m = _fingertip_standoff(self._parts)

    def draw(
        self,
        cv2,
        frame: np.ndarray,
        palm_m: np.ndarray,
        rotation: np.ndarray,
        opening_distance_m: float,
        opening_width: float,
        camera_from_left: np.ndarray,
        intrinsics: np.ndarray,
    ) -> None:
        """Rasterize the gripper. ``rotation`` columns are closing, up, approach."""

        closing = rotation[:, 0]
        approach = rotation[:, 2]
        mesh_rotation = np.column_stack((closing, approach, -rotation[:, 1]))
        origin = palm_m - approach * self._standoff_m
        pose = np.eye(4)
        pose[:3, :3] = mesh_rotation
        pose[:3, 3] = origin
        joint_q = _joint_position(opening_distance_m, opening_width)
        link_poses = _link_poses(joint_q)
        triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
        for part in self._parts:
            link_pose = pose @ link_poses[part["role"]]
            _collect_triangles(part, link_pose, camera_from_left, intrinsics, frame.shape, triangles)
        triangles.sort(key=lambda item: item[0], reverse=True)
        for _, pixels, color in triangles:
            cv2.fillConvexPoly(frame, pixels, color, cv2.LINE_AA)


def _joint_position(opening_distance_m: float, opening_width: float) -> float:
    """Map a pinch width onto the Dex1 prismatic joint. Closed is the upper stop."""

    if np.isfinite(opening_width):
        joint_q = _CLOSED_Q + float(opening_width) * (_OPEN_Q - _CLOSED_Q)
    else:
        joint_q = (_ZERO_GAP_M - float(opening_distance_m)) / 2.0
    return float(np.clip(joint_q, _OPEN_Q, _CLOSED_Q))


def _link_poses(joint_q: float) -> dict[str, np.ndarray]:
    finger1 = _translation((0.04055 - joint_q, 0.0593, 0.0152))
    finger1_2 = finger1 @ _translation((-0.013, 0.013, 0.0))
    finger1_3 = finger1_2 @ _translation((-0.0025165, 0.025043, -0.001))
    finger2 = _translation((-0.04055 + joint_q, 0.0593, -0.0152))
    finger2_2 = finger2 @ _translation((0.013, 0.013, 0.0))
    finger2_3 = finger2_2 @ _translation((0.0025165, 0.025043, 0.0294))
    return {
        "base": np.eye(4),
        "finger1": finger1,
        "finger1_2": finger1_2,
        "finger1_3": finger1_3,
        "finger2": finger2,
        "finger2_2": finger2_2,
        "finger2_3": finger2_3,
    }


def _fingertip_standoff(parts: list[dict[str, object]]) -> float:
    """Y coordinate of the fingertip meshes at the neutral joint position."""

    poses = _link_poses(0.0)
    samples = []
    for part in parts:
        if part["role"] not in {"finger1_3", "finger2_3"}:
            continue
        vertices = part["vertices"]
        transformed = vertices @ poses[part["role"]][:3, :3].T + poses[part["role"]][:3, 3]
        samples.append(transformed[:, 1].mean())
    return float(np.mean(samples))


def _collect_triangles(part, link_pose, camera_from_left, intrinsics, image_shape, triangles) -> None:
    rotation = link_pose[:3, :3]
    translation = link_pose[:3, 3]
    world = part["vertices"] @ rotation.T + translation
    camera = (camera_from_left[:3, :3] @ world.T).T + camera_from_left[:3, 3]
    depth = camera[:, 2]
    projected = (intrinsics @ camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    height_px, width_px = image_shape[:2]
    faces = part["faces"]
    color = part["color_bgr"]
    for face in faces:
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
        triangles.append((float(depth[face].mean()), face_pixels, color))


def _load_part(filename: str, rgb: tuple[float, float, float], role: str) -> dict[str, object]:
    vertices = _load_stl_vertices(MESH_DIR / filename)
    stride = _BASE_STRIDE if role == "base" else _FINGER_ROOT_STRIDE if role in {"finger1", "finger2"} else 1
    faces = np.arange(0, len(vertices) - 2, 3 * stride).reshape(-1, 1) + np.array([0, 1, 2])
    faces = faces[faces[:, 2] < len(vertices)]
    color = tuple(int(round(channel * 255)) for channel in (rgb[2], rgb[1], rgb[0]))
    return {"vertices": vertices, "faces": faces, "color_bgr": color, "role": role}


def _load_stl_vertices(path: Path) -> np.ndarray:
    """Read triangle vertices from a binary STL."""

    payload = path.read_bytes()
    triangle_count = int.from_bytes(payload[80:84], "little")
    records = np.frombuffer(payload, dtype=np.uint8, offset=84, count=triangle_count * 50).reshape(triangle_count, 50)
    coordinates = np.ascontiguousarray(records[:, 12:48]).view("<f4").reshape(triangle_count * 3, 3)
    return coordinates.astype(np.float64)


def _translation(xyz: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = xyz
    return transform
