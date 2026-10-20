"""G1 arm kinematics for one independently placed shoulder.

The base frame is the shoulder joint origin. The torso offset that locates the
shoulder on the body is removed, because module 8.2 places each arm base on
its own. Reach is the sum of the remaining link lengths, including the fixed
wrist-to-jaw offset. This module does not search bases or call a solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np

from se3 import pose_from_xyz_rpy, rotation_about_axis, transform


_ARM_JOINTS = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)
_FINGERTIP_LINKS = ("Link1_3", "Link2_3")


@dataclass(frozen=True)
class ArmJoint:
    """One revolute joint, with its URDF origin expressed in the parent link.

    ``origin`` is the 4x4 parent-to-joint transform at angle zero. Translation
    is meters and the moving axis is a unit vector in the joint frame.
    """

    name: str
    child_link: str
    origin: np.ndarray
    axis_xyz: np.ndarray
    lower_rad: float
    upper_rad: float
    mesh_file: str


@dataclass(frozen=True)
class ArmModel:
    """One 7-DoF G1 arm and the fixed transform from its wrist to the jaw pose.

    ``reach_m`` is the kinematic length used by the paper's ``r``. The jaw
    frame matches module 7: columns are closing, jaw normal, and approach.
    """

    side: str
    joints: tuple[ArmJoint, ...]
    wrist_from_jaw: np.ndarray
    reach_m: float
    urdf_path: Path

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("Arm side must be 'left' or 'right'.")
        if len(self.joints) != 7:
            raise ValueError("A G1 arm model requires seven revolute joints.")
        if self.reach_m <= 0.0:
            raise ValueError("Arm reach must be positive.")
        if np.asarray(self.wrist_from_jaw).shape != (4, 4):
            raise ValueError("wrist_from_jaw must be a 4x4 transform.")

    @classmethod
    def from_urdf(cls, urdf_path: Path, side: str) -> "ArmModel":
        """Read one arm from the G1 URDF.

        Args:
            urdf_path: ``g1_29dof_with_dex1_1.urdf``. Mesh paths inside it are
                relative to this file's directory.
            side: ``left`` or ``right``.
        Returns:
            An arm whose base sits on the shoulder joint, not on the torso.
        """

        if side not in {"left", "right"}:
            raise ValueError("Arm side must be 'left' or 'right'.")
        root = ElementTree.parse(urdf_path).getroot()
        joints = tuple(_arm_joints(root, side, urdf_path))
        wrist_from_jaw = _wrist_from_jaw(root, side, urdf_path)
        reach_m = _kinematic_reach(joints, wrist_from_jaw)
        return cls(
            side=side,
            joints=joints,
            wrist_from_jaw=wrist_from_jaw,
            reach_m=reach_m,
            urdf_path=urdf_path,
        )

    def nominal_joint_rad(self) -> np.ndarray:
        """Mid-range seed, in radians, shape ``(7,)``."""

        return np.array([0.5 * (joint.lower_rad + joint.upper_rad) for joint in self.joints], dtype=float)

    def clip_joint_rad(self, joint_rad: np.ndarray) -> np.ndarray:
        """Clamp seven joint angles to the URDF limits."""

        angles = _require_joints(joint_rad)
        lower = np.array([joint.lower_rad for joint in self.joints], dtype=float)
        upper = np.array([joint.upper_rad for joint in self.joints], dtype=float)
        return np.clip(angles, lower, upper)

    def within_limits(self, joint_rad: np.ndarray, tolerance_rad: float = 1e-6) -> bool:
        """Return whether every joint sits inside its URDF limit."""

        angles = _require_joints(joint_rad)
        lower = np.array([joint.lower_rad for joint in self.joints], dtype=float)
        upper = np.array([joint.upper_rad for joint in self.joints], dtype=float)
        return bool(np.all(angles >= lower - tolerance_rad) and np.all(angles <= upper + tolerance_rad))

    def forward(self, joint_rad: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Map joint angles to the jaw pose and the visual link poses.

        Args:
            joint_rad: Seven revolute angles in radians, shoulder pitch first.
        Returns:
            The 4x4 jaw pose in the arm base, and each visual link's 4x4 pose
            in that same base. Both are meters.
        """

        angles = _require_joints(joint_rad)
        pose = np.eye(4)
        links: dict[str, np.ndarray] = {}
        for joint, angle in zip(self.joints, angles):
            local = joint.origin.copy()
            local[:3, :3] = joint.origin[:3, :3] @ rotation_about_axis(joint.axis_xyz, float(angle))
            pose = pose @ local
            links[joint.child_link] = pose
        jaw = pose @ self.wrist_from_jaw
        return jaw, links

    def mjcf(self) -> str:
        """MuJoCo description of this arm, used only by the IK backend."""

        lines = [
            "<mujoco model=\"g1_arm\">",
            "  <compiler angle=\"radian\"/>",
            "  <worldbody>",
            "    <body name=\"base\">",
        ]
        pad = "      "
        for joint in self.joints:
            # MuJoCo rejects a moving body that has no inertia. The values only
            # keep the kinematic model loadable; IK does not use the mass.
            lines.append(
                f"{pad}<body name=\"{joint.child_link}\" pos=\"{_numbers(joint.origin[:3, 3])}\" "
                f"quat=\"{_numbers(_wxyz(joint.origin[:3, :3]))}\">"
            )
            lines.append(f"{pad}  <inertial pos=\"0 0 0\" mass=\"1\" diaginertia=\"0.01 0.01 0.01\"/>")
            lines.append(
                f"{pad}  <joint name=\"{joint.name}\" type=\"hinge\" axis=\"{_numbers(joint.axis_xyz)}\" "
                f"range=\"{joint.lower_rad:.8g} {joint.upper_rad:.8g}\"/>"
            )
            pad += "  "
        lines.append(
            f"{pad}<body name=\"jaw\" pos=\"{_numbers(self.wrist_from_jaw[:3, 3])}\" "
            f"quat=\"{_numbers(_wxyz(self.wrist_from_jaw[:3, :3]))}\"/>"
        )
        for _joint in self.joints:
            pad = pad[:-2]
            lines.append(f"{pad}</body>")
        lines.extend(["    </body>", "  </worldbody>", "</mujoco>"])
        return "\n".join(lines)


def _arm_joints(root: ElementTree.Element, side: str, urdf_path: Path) -> list[ArmJoint]:
    joints = []
    for suffix in _ARM_JOINTS:
        element = _joint(root, f"{side}_{suffix}")
        origin = _origin(element)
        if suffix == "shoulder_pitch_joint":
            origin = origin.copy()
            origin[:3, 3] = 0.0
        axis = np.array([float(value) for value in element.find("axis").attrib["xyz"].split()], dtype=float)
        axis = axis / np.linalg.norm(axis)
        limit = element.find("limit")
        child = element.find("child").attrib["link"]
        joints.append(
            ArmJoint(
                name=element.attrib["name"],
                child_link=child,
                origin=origin,
                axis_xyz=axis,
                lower_rad=float(limit.attrib["lower"]),
                upper_rad=float(limit.attrib["upper"]),
                mesh_file=_mesh_file(root, child, urdf_path),
            )
        )
    return joints


def _wrist_from_jaw(root: ElementTree.Element, side: str, urdf_path: Path) -> np.ndarray:
    mount = _origin(_joint(root, f"{side}_dex1_mount_joint"))
    standoff_m = _fingertip_standoff_m(root, urdf_path)
    dex1_from_jaw = transform(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float),
        np.array([0.0, standoff_m, 0.0], dtype=float),
    )
    return mount @ dex1_from_jaw


def _kinematic_reach(joints: tuple[ArmJoint, ...] | list[ArmJoint], wrist_from_jaw: np.ndarray) -> float:
    length = sum(float(np.linalg.norm(joint.origin[:3, 3])) for joint in joints)
    length += float(np.linalg.norm(wrist_from_jaw[:3, 3]))
    return length


def _fingertip_standoff_m(root: ElementTree.Element, urdf_path: Path) -> float:
    """Mean fingertip Y in the Dex1 base, matching the gripper drawer's neutral pose."""

    samples = []
    for suffix in _FINGERTIP_LINKS:
        link_name = f"left_dex1_{suffix}"
        pose = _fixed_pose(root, "left_dex1_base_link", link_name)
        vertices = _stl_vertices(Path(_mesh_file(root, link_name, urdf_path)))
        placed = vertices @ pose[:3, :3].T + pose[:3, 3]
        samples.append(float(placed[:, 1].mean()))
    return float(np.mean(samples))


def _fixed_pose(root: ElementTree.Element, base_link: str, target_link: str) -> np.ndarray:
    """Pose of ``target_link`` in ``base_link`` with every moving joint at zero."""

    parent_of: dict[str, str] = {}
    origin_of: dict[str, np.ndarray] = {}
    pending = [base_link]
    seen = {base_link}
    while pending:
        link = pending.pop(0)
        if link == target_link:
            break
        for element in root.findall("joint"):
            if element.find("parent").attrib["link"] != link:
                continue
            child = element.find("child").attrib["link"]
            if child in seen:
                continue
            seen.add(child)
            parent_of[child] = link
            origin_of[child] = _origin(element)
            pending.append(child)
    if target_link not in seen:
        raise ValueError(f"URDF has no path from {base_link} to {target_link}.")
    locals_from_base: list[np.ndarray] = []
    link = target_link
    while link != base_link:
        locals_from_base.append(origin_of[link])
        link = parent_of[link]
    pose = np.eye(4)
    for local in reversed(locals_from_base):
        pose = pose @ local
    return pose


def _joint(root: ElementTree.Element, name: str) -> ElementTree.Element:
    for element in root.findall("joint"):
        if element.attrib.get("name") == name:
            return element
    raise ValueError(f"URDF is missing joint {name}.")


def _origin(element: ElementTree.Element) -> np.ndarray:
    origin = element.find("origin")
    if origin is None:
        return np.eye(4)
    xyz = np.array([float(value) for value in origin.attrib.get("xyz", "0 0 0").split()], dtype=float)
    rpy = np.array([float(value) for value in origin.attrib.get("rpy", "0 0 0").split()], dtype=float)
    return pose_from_xyz_rpy(xyz, rpy)


def _mesh_file(root: ElementTree.Element, link_name: str, urdf_path: Path) -> str:
    for link in root.findall("link"):
        if link.attrib.get("name") != link_name:
            continue
        visual = link.find("visual")
        mesh = None if visual is None else visual.find("./geometry/mesh")
        if mesh is None:
            raise ValueError(f"URDF link {link_name} has no visual mesh.")
        return str((urdf_path.parent / mesh.attrib["filename"]).resolve())
    raise ValueError(f"URDF is missing link {link_name}.")


def _stl_vertices(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    triangle_count = int.from_bytes(payload[80:84], "little")
    records = np.frombuffer(payload, dtype=np.uint8, offset=84, count=triangle_count * 50).reshape(triangle_count, 50)
    coordinates = np.ascontiguousarray(records[:, 12:48]).view("<f4").reshape(triangle_count * 3, 3)
    return coordinates.astype(np.float64)


def _numbers(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = _quaternion(rotation)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    from se3 import quaternion_xyzw

    return quaternion_xyzw(rotation)


def _require_joints(joint_rad: np.ndarray) -> np.ndarray:
    angles = np.asarray(joint_rad, dtype=float)
    if angles.shape != (7,):
        raise ValueError(f"joint_rad must have shape (7,), received {angles.shape}.")
    if not np.all(np.isfinite(angles)):
        raise ValueError("joint_rad must be finite.")
    return angles
