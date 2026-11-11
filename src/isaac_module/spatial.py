"""Quaternion <-> Viam orientation-vector conversions.

Ported from rdk/spatialmath (QuatToOV in quaternion.go and
OrientationVector.Quaternion() in orientationVector.go) so that poses
reported by this module match what the rest of the Viam ecosystem expects.

Quaternions here are (w, x, y, z) tuples. Isaac Sim's core API also uses
w-first quaternions, so no reordering is needed at the call sites.
"""

import math
from typing import Sequence, Tuple

_ANGLE_EPSILON = 1e-4
_POLE_RADIUS = 1e-4

Quat = Tuple[float, float, float, float]
Vec3 = Tuple[float, float, float]


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    p = (0.0, v[0], v[1], v[2])
    r = quat_mul(quat_mul(q, p), quat_conj(q))
    return (r[1], r[2], r[3])


def pose_in_frame(
    frame_position: Vec3, frame_rotation: Quat, position: Vec3, rotation: Quat
) -> Tuple[Vec3, Quat]:
    """Express a world pose in the frame whose world pose is given."""
    inverse = quat_conj(frame_rotation)
    offset = (
        position[0] - frame_position[0],
        position[1] - frame_position[1],
        position[2] - frame_position[2],
    )
    return quat_rotate(inverse, offset), quat_mul(inverse, rotation)


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quat:
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(angle / 2.0) / n
    return (math.cos(angle / 2.0), x * s, y * s, z * s)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def quat_to_ov(q: Quat) -> Tuple[float, float, float, float]:
    """Convert a (w,x,y,z) quaternion to a Viam orientation vector
    (ox, oy, oz, theta) with theta in radians. Mirrors rdk's QuatToOV."""
    # rdk uses the -X axis here, not +X.
    new_x = quat_rotate(q, (-1.0, 0.0, 0.0))
    new_z = quat_rotate(q, (0.0, 0.0, 1.0))
    ox, oy, oz = new_z
    theta: float

    if 1 - abs(new_z[2]) > _POLE_RADIUS:
        v1 = new_z
        v2 = new_x
        # normal to the local-x, local-z, origin plane
        norm1 = _cross(v1, v2)
        # normal to the global-z, local-z, origin plane
        norm2 = _cross(v1, (0.0, 0.0, 1.0))
        cos_theta = _dot(norm1, norm2) / (_norm(norm1) * _norm(norm2))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)
        if theta > _POLE_RADIUS:
            # acos is always positive; determine the sign by rotating newZ by
            # -theta about the OV axis and checking if we land coplanar with
            # local-x, global-z, origin.
            q2 = quat_from_axis_angle((ox, oy, oz), -theta)
            test_z = quat_rotate(q2, (0.0, 0.0, 1.0))
            norm3 = _cross(v1, test_z)
            cos_test = _dot(norm1, norm3) / (_norm(norm1) * _norm(norm3))
            if 1 - cos_test < _ANGLE_EPSILON * _ANGLE_EPSILON:
                theta = -theta
        else:
            theta = 0.0
    else:
        # pointing (nearly) straight along +/- Z
        if new_z[2] >= 0:
            theta = -math.atan2(new_x[1], -new_x[0])
        else:
            theta = -math.atan2(new_x[1], new_x[0])

    if theta == -0.0:
        theta = 0.0
    return (ox, oy, oz, theta)


def ov_to_quat(ox: float, oy: float, oz: float, theta: float) -> Quat:
    """Convert a Viam orientation vector (theta in radians) to a (w,x,y,z)
    quaternion. Mirrors rdk's OrientationVector.Quaternion() (ZYZ order)."""
    n = math.sqrt(ox * ox + oy * oy + oz * oz)
    if n == 0:
        raise ValueError("orientation vector has zero length")
    ox, oy, oz = ox / n, oy / n, oz / n

    lat = math.acos(max(-1.0, min(1.0, oz)))
    lon = 0.0
    if 1 - abs(oz) > _ANGLE_EPSILON:
        lon = math.atan2(oy, ox)

    # ZYZ: Rz(lon) * Ry(lat) * Rz(theta)
    rz1 = (math.cos(lon / 2), 0.0, 0.0, math.sin(lon / 2))
    ry = (math.cos(lat / 2), 0.0, math.sin(lat / 2), 0.0)
    rz2 = (math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2))
    return quat_mul(quat_mul(rz1, ry), rz2)


def quat_from_euler_deg(roll: float, pitch: float, yaw: float) -> Quat:
    """XYZ-extrinsic (roll about X, pitch about Y, yaw about Z) euler degrees
    to a (w,x,y,z) quaternion. Used for placing cameras/props from config."""
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    qx = (math.cos(r / 2), math.sin(r / 2), 0.0, 0.0)
    qy = (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)
    qz = (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))
    return quat_mul(qz, quat_mul(qy, qx))


def look_at_quat(position: Vec3, target: Vec3) -> Quat:
    """Quaternion (w,x,y,z) in isaac's world camera convention (+X forward,
    +Z up) pointing from position toward target."""
    fx = target[0] - position[0]
    fy = target[1] - position[1]
    fz = target[2] - position[2]
    n = math.sqrt(fx * fx + fy * fy + fz * fz)
    if n == 0:
        return (1.0, 0.0, 0.0, 0.0)
    fx, fy, fz = fx / n, fy / n, fz / n
    yaw = math.atan2(fy, fx)
    pitch = -math.asin(max(-1.0, min(1.0, fz)))
    qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    qy = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
    return quat_mul(qz, qy)


def orbit_to_position(
    target: Sequence[float], distance: float, yaw_deg: float, pitch_deg: float
) -> Vec3:
    """Camera position on a sphere around target. Z-up; yaw is measured
    about +Z from +X; pitch is elevation above the XY plane."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    horizontal = distance * math.cos(pitch)
    return (
        float(target[0]) + horizontal * math.cos(yaw),
        float(target[1]) + horizontal * math.sin(yaw),
        float(target[2]) + distance * math.sin(pitch),
    )


def position_to_orbit(
    position: Sequence[float], target: Sequence[float]
) -> Tuple[float, float, float]:
    """Inverse of orbit_to_position: (distance, yaw_deg, pitch_deg)."""
    dx = float(position[0]) - float(target[0])
    dy = float(position[1]) - float(target[1])
    dz = float(position[2]) - float(target[2])
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance == 0.0:
        raise ValueError("camera position and target must differ")
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, dz / distance))))
    return distance, yaw, pitch


def to_vec3(value: Sequence[float], default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if value is None:
        return default
    vals = [float(v) for v in value]
    if len(vals) != 3:
        raise ValueError(f"expected 3 values, got {len(vals)}")
    return (vals[0], vals[1], vals[2])
