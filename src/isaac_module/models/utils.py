import math
from typing import Any, Dict, Optional, Sequence, Tuple

from viam.proto.app.robot import ComponentConfig
from viam.utils import struct_to_dict

from ..spatial import (
    Quat,
    Vec3,
    ov_to_quat,
    quat_from_axis_angle,
    quat_from_euler_deg,
    quat_mul,
    quat_rotate,
)


def get_attrs(config: ComponentConfig) -> Dict[str, Any]:
    return struct_to_dict(config.attributes)


def frame_pose(config: ComponentConfig) -> Tuple[Optional[Vec3], Optional[Quat]]:
    """Spawn pose from the component's standard frame config: position in
    meters (frame translations are mm) and a (w,x,y,z) quaternion, or None
    for whatever isn't set. The frame system is the preferred way to place
    isaac-sim components - it keeps viam's view of the machine (e.g. the
    motion service) consistent with where prims actually are in the sim."""
    if not config.HasField("frame"):
        return None, None
    frame = config.frame
    t = frame.translation
    position = (t.x / 1000.0, t.y / 1000.0, t.z / 1000.0)

    quat: Optional[Quat] = None
    o = frame.orientation
    which = o.WhichOneof("type")
    if which == "quaternion":
        q = o.quaternion
        quat = (q.w, q.x, q.y, q.z)
    elif which == "vector_radians":
        v = o.vector_radians
        quat = ov_to_quat(v.x, v.y, v.z, v.theta)
    elif which == "vector_degrees":
        v = o.vector_degrees
        quat = ov_to_quat(v.x, v.y, v.z, math.radians(v.theta))
    elif which == "euler_angles":
        e = o.euler_angles  # radians, applied as Rz(yaw)*Ry(pitch)*Rx(roll)
        qz = (math.cos(e.yaw / 2), 0.0, 0.0, math.sin(e.yaw / 2))
        qy = (math.cos(e.pitch / 2), 0.0, math.sin(e.pitch / 2), 0.0)
        qx = (math.cos(e.roll / 2), math.sin(e.roll / 2), 0.0, 0.0)
        quat = quat_mul(qz, quat_mul(qy, qx))
    elif which == "axis_angles":
        a = o.axis_angles
        quat = quat_from_axis_angle((a.x, a.y, a.z), a.theta)
    return position, quat


def camera_frame_orientation(config: ComponentConfig) -> Optional[Quat]:
    """Frame orientation for an unparented camera; viam-server fills in an
    identity quaternion when the user set none, so identity means unset."""
    quat = frame_pose(config)[1]
    if quat is None or abs(abs(quat[0]) - 1) < 1e-9:
        return None
    return quat


def apply_frame_to_attrs(
    config: ComponentConfig, attrs: Dict[str, Any]
) -> Dict[str, Any]:
    """Fold the frame config into the spawn attrs (frame wins over the
    legacy position/orientation attributes)."""
    position, quat = frame_pose(config)
    if position is not None:
        attrs["position"] = list(position)
    if quat is not None:
        attrs["orientation_wxyz"] = list(quat)
    return attrs


def apply_camera_frame_to_attrs(
    config: ComponentConfig, attrs: Dict[str, Any]
) -> Dict[str, Any]:
    if (
        attrs.get("parent_prim")
        and config.HasField("frame")
        and config.frame.parent != "world"
    ):
        position, quat = frame_pose(config)
        attrs.pop("local_position", None)
        attrs.pop("local_orientation_rpy_deg", None)
        # The Viam parent frame can sit rotated inside the prim (e.g. an arm's
        # end frame vs its USD flange link); rotate the frame's pose into the
        # prim's coordinates so the sim and the frame system agree.
        offset = attrs.pop("parent_prim_rpy_deg", None)
        if offset is not None:
            r = quat_from_euler_deg(*(float(v) for v in offset))
            if position is not None:
                position = quat_rotate(r, position)
            quat = quat_mul(r, quat if quat is not None else (1.0, 0.0, 0.0, 0.0))
        if position is not None:
            attrs["local_position"] = list(position)
        if quat is not None:
            attrs["local_orientation_wxyz"] = list(quat)
        attrs["local_camera_axes"] = "ros"
        return attrs
    position, _ = frame_pose(config)
    quat = camera_frame_orientation(config)
    if position is not None:
        attrs["position"] = list(position)
    # Viam camera frames have Z out of the lens (optical), like real cameras;
    # target / legacy orientations keep Isaac "world" camera axes.
    if quat is not None:
        attrs["orientation_wxyz"] = list(quat)
        attrs["camera_axes"] = "ros"
    return attrs


def apply_lidar_frame_to_attrs(
    config: ComponentConfig, attrs: Dict[str, Any]
) -> Dict[str, Any]:
    """Lidars use Isaac-native axes (X forward, Z up) everywhere; with a
    parent_prim the frame is the pose relative to that prim."""
    position, quat = frame_pose(config)
    prefix = "local_" if attrs.get("parent_prim") else ""
    if position is not None:
        attrs[f"{prefix}position"] = list(position)
    if quat is not None:
        attrs[f"{prefix}orientation_wxyz"] = list(quat)
    return attrs


def validate_sim_component(
    config: ComponentConfig, needs_source: bool = True
) -> Tuple[Sequence[str], Sequence[str]]:
    """Shared validation for arm/camera/base: they must name their world
    component (so viam-server starts it first) and, when they spawn a prim,
    say what to spawn."""
    attrs = struct_to_dict(config.attributes)
    world = attrs.get("world")
    if not world or not isinstance(world, str):
        raise ValueError(
            f'{config.name}: set the "world" attribute to the name of your '
            "viam-labs:isaac-sim:world component"
        )
    if needs_source and not (
        attrs.get("asset") or attrs.get("usd_path") or attrs.get("prim_path")
    ):
        raise ValueError(
            f'{config.name}: set "asset" (e.g. "ur20"), "usd_path", or '
            '"prim_path" (to attach to something already in the stage)'
        )
    return [world], []
