import math

import pytest

from isaac_module.spatial import (
    look_at_quat,
    orbit_to_position,
    position_to_orbit,
    ov_to_quat,
    quat_from_axis_angle,
    quat_from_euler_deg,
    quat_mul,
    quat_rotate,
    quat_to_ov,
)


def test_identity_quat():
    ox, oy, oz, theta = quat_to_ov((1.0, 0.0, 0.0, 0.0))
    assert (ox, oy, oz) == pytest.approx((0.0, 0.0, 1.0))
    assert theta == pytest.approx(0.0)


def test_flip_about_x():
    # 180 deg about X: z-axis points down
    q = quat_from_axis_angle((1, 0, 0), math.pi)
    ox, oy, oz, theta = quat_to_ov(q)
    assert (ox, oy, oz) == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


def test_rot90_about_x_points_neg_y():
    # +90 deg about X sends +Z to -Y... z-hat -> (0,-1,0)? rotating (0,0,1)
    # by +90 about x: (0, -1*... ) -> (0, -sin90, cos90) = (0,-1,0)
    q = quat_from_axis_angle((1, 0, 0), math.pi / 2)
    ox, oy, oz, _ = quat_to_ov(q)
    assert (ox, oy, oz) == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)


@pytest.mark.parametrize(
    "ov",
    [
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -0.5),
        (0.5, 0.5, 0.707, 0.2),
        (-0.3, 0.8, -0.5, 2.5),
        (0.0, 0.0, -1.0, 0.3),
        (0.1, -0.9, 0.4, -2.8),
    ],
)
def test_round_trip_ov(ov):
    ox, oy, oz, theta = ov
    n = math.sqrt(ox * ox + oy * oy + oz * oz)
    ox, oy, oz = ox / n, oy / n, oz / n
    q = ov_to_quat(ox, oy, oz, theta)
    ox2, oy2, oz2, theta2 = quat_to_ov(q)
    assert (ox2, oy2, oz2) == pytest.approx((ox, oy, oz), abs=1e-6)
    # theta is only well-defined modulo 2*pi
    dt = (theta2 - theta + math.pi) % (2 * math.pi) - math.pi
    assert dt == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "q",
    [
        quat_from_euler_deg(10, 20, 30),
        quat_from_euler_deg(-45, 80, 170),
        quat_from_euler_deg(90, 0, 0),
        quat_from_axis_angle((1, 1, 1), 1.0),
        quat_from_axis_angle((0, 1, 0), -2.0),
    ],
)
def test_round_trip_quat(q):
    """quat -> OV -> quat must represent the same rotation (q or -q)."""
    ox, oy, oz, theta = quat_to_ov(q)
    q2 = ov_to_quat(ox, oy, oz, theta)
    # same rotation iff |dot| == 1
    dot = sum(a * b for a, b in zip(q, q2))
    assert abs(dot) == pytest.approx(1.0, abs=1e-6)


def test_rotate_helper():
    q = quat_from_axis_angle((0, 0, 1), math.pi / 2)
    v = quat_rotate(q, (1.0, 0.0, 0.0))
    assert v == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_mul_identity():
    q = quat_from_euler_deg(11, 22, 33)
    assert quat_mul(q, (1, 0, 0, 0)) == pytest.approx(q)


def test_look_at_points_at_target():
    # camera at (3,3,3) looking at origin: +X (isaac world-camera forward)
    # should map to the normalized direction toward the target
    q = look_at_quat((3.0, 3.0, 3.0), (0.0, 0.0, 0.0))
    fwd = quat_rotate(q, (1.0, 0.0, 0.0))
    n = (3.0**2 * 3) ** 0.5
    assert fwd == pytest.approx((-3.0 / n, -3.0 / n, -3.0 / n), abs=1e-9)


def test_look_at_straight_down():
    q = look_at_quat((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    fwd = quat_rotate(q, (1.0, 0.0, 0.0))
    assert fwd[2] == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.parametrize(
    "yaw, pitch",
    [(0.0, 0.0), (90.0, 0.0), (-135.0, 30.0), (45.0, -60.0), (179.0, 89.0)],
)
def test_orbit_round_trip(yaw, pitch):
    target = (0.5, -0.25, 0.75)
    position = orbit_to_position(target, 2.0, yaw, pitch)
    distance, yaw_out, pitch_out = position_to_orbit(position, target)
    assert distance == pytest.approx(2.0)
    assert yaw_out == pytest.approx(yaw, abs=1e-9)
    assert pitch_out == pytest.approx(pitch, abs=1e-9)


def test_orbit_axes_are_z_up_yaw_from_x():
    assert orbit_to_position((0, 0, 0), 3.0, 0.0, 0.0) == pytest.approx((3.0, 0.0, 0.0))
    assert orbit_to_position((0, 0, 0), 3.0, 90.0, 0.0) == pytest.approx((0.0, 3.0, 0.0), abs=1e-12)
    assert orbit_to_position((0, 0, 1), 2.0, 0.0, 90.0) == pytest.approx((0.0, 0.0, 3.0), abs=1e-12)


def test_look_at_from_orbit_points_at_target():
    target = (0.2, 0.1, 0.5)
    position = orbit_to_position(target, 1.5, 30.0, 20.0)
    forward = quat_rotate(look_at_quat(position, target), (1.0, 0.0, 0.0))
    expected = [t - p for t, p in zip(target, position)]
    norm = math.sqrt(sum(v * v for v in expected))
    assert forward == pytest.approx(tuple(v / norm for v in expected), abs=1e-9)


def test_position_to_orbit_rejects_coincident_points():
    with pytest.raises(ValueError, match="must differ"):
        position_to_orbit((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
