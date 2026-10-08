import math
from typing import Any, Dict, Optional, Sequence, Tuple

from viam.proto.app.robot import ComponentConfig
from viam.utils import struct_to_dict

from ..spatial import Quat, Vec3, ov_to_quat, quat_from_axis_angle, quat_mul


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


def apply_frame_to_attrs(config: ComponentConfig, attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the frame config into the spawn attrs (frame wins over the
    legacy position/orientation attributes)."""
    position, quat = frame_pose(config)
    if position is not None:
        attrs["position"] = list(position)
    if quat is not None:
        attrs["orientation_wxyz"] = list(quat)
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
