"""The floating-body force model and the motor / movement_sensor models,
exercised against the mock backend (the same force model Isaac uses, minus
PhysX) and against a fake RigidPrim view for the Isaac body handle."""

import asyncio
import math
import threading
import time
from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.bodies import (
    GRAVITY_MPS2,
    BuoyancySpec,
    GeoOrigin,
    MovementSensorSpec,
    ThrusterState,
    compass_heading_deg,
    net_wrench,
    quat_from_yaw,
    quat_to_rpy,
)
from isaac_module.models.motor import IsaacMotor
from isaac_module.models.movement_sensor import IsaacMovementSensor
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import IsaacBodyHandle, SimConfig, SimManager

IDENTITY = (1.0, 0.0, 0.0, 0.0)
STILL = (0.0, 0.0, 0.0)
# a 1.0 x 0.6 x 0.3 m, 20 kg box boat
BOAT = {"name": "boat", "type": "cube", "size": 1.0, "scale": [1.0, 0.6, 0.3], "mass": 20}


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _spec(**overrides) -> BuoyancySpec:
    prop = dict(BOAT, buoyancy=overrides or True)
    return BuoyancySpec.from_prop(prop, water_level_m=0.0)


# ----------------------------------------------------------------------
# BuoyancySpec
# ----------------------------------------------------------------------


def test_buoyancy_spec_from_a_cube_prop():
    spec = _spec()
    assert spec.size_m == (1.0, 0.6, 0.3)
    assert spec.mass_kg == 20.0
    assert spec.waterplane_area_m2 == pytest.approx(0.6)
    assert spec.draft_m == pytest.approx(20.0 / (1000.0 * 0.6))
    assert spec.rest_z_m == pytest.approx(0.15 - spec.draft_m)
    assert spec.bottom_z_m == pytest.approx(-0.15)
    assert len(spec.points) == 4


def test_buoyancy_spec_is_none_without_the_block():
    assert BuoyancySpec.from_prop(dict(BOAT), 0.0) is None
    assert BuoyancySpec.from_prop(dict(BOAT, buoyancy=False), 0.0) is None


def test_buoyancy_spec_accepts_a_catamaran_style_block():
    spec = _spec(
        waterplane_area_m2=0.3,
        points=[[-0.33, 0.23], [0, 0.23], [0.33, 0.23], [-0.33, -0.23], [0, -0.23], [0.33, -0.23]],
        linear_damping=[1, 10, 100],
    )
    assert spec.waterplane_area_m2 == 0.3
    assert len(spec.points) == 6
    assert spec.linear_damping == (1.0, 10.0, 100.0)


def test_buoyancy_spec_for_a_usd_prop_needs_size():
    usd = {"name": "boat", "type": "usd", "usd_path": "x.usd", "mass": 5, "buoyancy": True}
    with pytest.raises(ValueError, match="size_m"):
        BuoyancySpec.from_prop(usd, 0.0)
    spec = BuoyancySpec.from_prop(dict(usd, buoyancy={"size_m": [0.5, 0.2, 0.1]}), 0.0)
    assert spec is not None and spec.size_m == (0.5, 0.2, 0.1)


@pytest.mark.parametrize(
    ("prop", "match"),
    (
        (dict(BOAT, buoyancy=True, fixed=True), "fixed"),
        ({k: v for k, v in dict(BOAT, buoyancy=True).items() if k != "mass"}, "mass"),
        (dict(BOAT, buoyancy=True, mass=10_000), "sink"),
        (dict(BOAT, buoyancy={"linear_damping": [-1, 0, 0]}), "linear_damping"),
        (dict(BOAT, buoyancy={"points": []}), "points"),
        (dict(BOAT, buoyancy="yes"), "buoyancy"),
    ),
)
def test_buoyancy_spec_rejects_bad_config(prop, match):
    with pytest.raises(ValueError, match=match):
        BuoyancySpec.from_prop(prop, 0.0)


# ----------------------------------------------------------------------
# net_wrench
# ----------------------------------------------------------------------


def test_buoyancy_balances_weight_at_rest_draft():
    spec = _spec()
    force, torque = net_wrench(spec, (0.0, 0.0, spec.rest_z_m), IDENTITY, STILL, STILL, [])
    assert force[2] == pytest.approx(spec.mass_kg * GRAVITY_MPS2, rel=1e-6)
    assert force[0] == force[1] == 0.0
    assert torque == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_buoyancy_restores_heave():
    spec = _spec()
    weight = spec.mass_kg * GRAVITY_MPS2
    low, _ = net_wrench(spec, (0.0, 0.0, spec.rest_z_m - 0.02), IDENTITY, STILL, STILL, [])
    high, _ = net_wrench(spec, (0.0, 0.0, spec.rest_z_m + 0.02), IDENTITY, STILL, STILL, [])
    assert low[2] > weight > high[2]
    # a body lifted clear of the water gets no lift at all
    dry, _ = net_wrench(spec, (0.0, 0.0, 5.0), IDENTITY, STILL, STILL, [])
    assert dry[2] == 0.0


def test_buoyancy_rights_a_rolled_body():
    spec = _spec()
    roll = math.radians(10.0)
    rolled = (math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0)  # +roll: left side down
    _, torque = net_wrench(spec, (0.0, 0.0, spec.rest_z_m), rolled, STILL, STILL, [])
    # the deeper (+Y) side gets more lift, which torques the body back
    assert torque[0] < 0.0
    assert abs(torque[1]) < 1e-9


def test_damping_opposes_motion_more_in_sway_than_surge():
    spec = _spec()
    pose = (0.0, 0.0, spec.rest_z_m)
    surge, _ = net_wrench(spec, pose, IDENTITY, (1.0, 0.0, 0.0), STILL, [])
    sway, _ = net_wrench(spec, pose, IDENTITY, (0.0, 1.0, 0.0), STILL, [])
    assert surge[0] < 0.0 and sway[1] < 0.0
    assert abs(sway[1]) > abs(surge[0])
    _, spin = net_wrench(spec, pose, IDENTITY, STILL, (0.0, 0.0, 1.0), [])
    assert spin[2] < 0.0


def test_thrusters_push_forward_and_differential_thrust_yaws():
    spec = _spec()
    left = ThrusterState("left", (-0.45, 0.25, -0.1), (1.0, 0.0, 0.0), force_n=10.0)
    right = ThrusterState("right", (-0.45, -0.25, -0.1), (1.0, 0.0, 0.0), force_n=10.0)
    pose = (0.0, 0.0, spec.rest_z_m)
    both, torque_both = net_wrench(spec, pose, IDENTITY, STILL, STILL, [left, right])
    assert both[0] == pytest.approx(20.0)
    assert torque_both[2] == pytest.approx(0.0, abs=1e-9)
    # left thruster alone: nose swings right, i.e. negative yaw torque
    _, torque_left = net_wrench(spec, pose, IDENTITY, STILL, STILL, [left])
    assert torque_left[2] == pytest.approx(-0.25 * 10.0)
    # the push follows the body's heading
    yawed = quat_from_yaw(math.pi / 2)
    turned, _ = net_wrench(spec, pose, yawed, STILL, STILL, [left, right])
    assert turned[0] == pytest.approx(0.0, abs=1e-9)
    assert turned[1] == pytest.approx(20.0)


def test_body_without_buoyancy_only_feels_its_thrusters():
    thruster = ThrusterState("t", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), force_n=3.0)
    force, torque = net_wrench(None, (0.0, 0.0, 1.0), IDENTITY, (5.0, 5.0, 5.0), STILL, [thruster])
    assert force == pytest.approx((0.0, 3.0, 0.0))
    assert torque == pytest.approx((0.0, 0.0, 0.0))


def test_compass_heading_is_clockwise_from_north():
    assert compass_heading_deg(IDENTITY) == pytest.approx(90.0)  # +X = east
    assert compass_heading_deg(quat_from_yaw(math.pi / 2)) == pytest.approx(0.0)  # north
    assert compass_heading_deg(quat_from_yaw(-math.pi / 2)) == pytest.approx(180.0)
    assert compass_heading_deg(quat_from_yaw(math.pi)) == pytest.approx(270.0)


def test_geo_origin_offsets():
    origin = GeoOrigin(40.7858, -73.9846, 2.0)
    lat, lon, alt = origin.to_geo(0.0, 0.0, 0.5)
    assert (lat, lon, alt) == pytest.approx((40.7858, -73.9846, 2.5))
    lat, lon, _ = origin.to_geo(0.0, 111_195.0, 0.0)  # one degree north
    assert lat == pytest.approx(41.7858, abs=1e-3)
    assert lon == pytest.approx(-73.9846)
    lat, lon, _ = origin.to_geo(100.0, 0.0, 0.0)  # east, scaled by cos(lat)
    assert lon > -73.9846
    assert lat == pytest.approx(40.7858)
    with pytest.raises(ValueError, match="origin"):
        GeoOrigin.from_attrs({"latitude": 95})


def test_movement_sensor_spec_rejects_negative_noise():
    with pytest.raises(ValueError, match="position_noise_m"):
        MovementSensorSpec.from_attrs({"position_noise_m": -1})


# ----------------------------------------------------------------------
# Mock end-to-end: world with a floating prop -> thrusters -> movement sensor
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    previous = SimManager._instance
    manager = SimManager()
    SimManager._instance = manager
    thread = threading.Thread(target=manager.main_loop, daemon=True)
    thread.start()
    yield manager
    manager.request_stop()
    thread.join(timeout=5)
    SimManager._instance = previous


@pytest.fixture(scope="module")
def world(sim):
    return IsaacWorld.new(
        _config(
            "sim-world",
            {
                "mock": True,
                "water": {"level_m": 0.0, "size_m": 200},
                "props": [
                    dict(BOAT, position=[0.0, 0.0, 0.1], buoyancy=True),
                    {
                        "name": "piling",
                        "type": "cube",
                        "fixed": True,
                        "position": [5.0, 5.0, 0.5],
                        "scale": [0.3, 0.3, 2.0],
                    },
                ],
            },
        ),
        {},
    )


@pytest.fixture(scope="module")
def boat(world):
    left = IsaacMotor.new(
        _config(
            "left-thruster",
            {
                "world": "sim-world",
                "prop": "boat",
                "position": [-0.45, 0.23, -0.1],
                "max_thrust_n": 49,
                "max_rpm": 3000,
            },
        ),
        {},
    )
    right = IsaacMotor.new(
        _config(
            "right-thruster",
            {
                "world": "sim-world",
                "prop": "boat",
                "position": [-0.45, -0.23, -0.1],
                "max_thrust_n": 49,
                "max_rpm": 3000,
            },
        ),
        {},
    )
    sensor = IsaacMovementSensor.new(
        _config(
            "nav",
            {
                "world": "sim-world",
                "prop": "boat",
                "origin": {"latitude": 40.7858, "longitude": -73.9846, "altitude_m": 0},
            },
        ),
        {},
    )
    return SimpleNamespace(world=world, left=left, right=right, sensor=sensor)


def _run(coro):
    return asyncio.run(coro)


def _park(boat) -> None:
    _run(boat.left.stop())
    _run(boat.right.stop())
    _run(boat.world.do_command({"command": "park_props", "props": ["boat"]}))
    time.sleep(0.05)


def test_world_water_is_recorded_in_config(sim, world):
    assert sim.cfg.water == {"level_m": 0.0, "size_m": 200}
    assert sim.cfg.water_level_m() == 0.0
    assert SimConfig().water_level_m() == 0.0


def test_floating_prop_sits_at_its_rest_draft(boat):
    _park(boat)
    readings = _run(boat.sensor.get_readings())
    spec = BuoyancySpec.from_prop(dict(BOAT, buoyancy=True), 0.0)
    assert readings["sim_position_m"]["z"] == pytest.approx(spec.rest_z_m)
    assert readings["speed_mps"] == pytest.approx(0.0, abs=1e-3)


def test_two_thrusters_move_the_boat_forward(boat):
    _park(boat)
    before = _run(boat.sensor.get_readings())

    _run(boat.left.set_power(1.0))
    _run(boat.right.set_power(1.0))
    powered, power = _run(boat.left.is_powered())
    assert powered and power == 1.0
    assert _run(boat.left.is_moving())
    time.sleep(1.0)
    after = _run(boat.sensor.get_readings())

    assert after["sim_position_m"]["x"] > before["sim_position_m"]["x"] + 0.5
    assert abs(after["sim_position_m"]["y"] - before["sim_position_m"]["y"]) < 0.05
    assert after["speed_mps"] > 1.0
    # heading stays east while both thrusters push equally
    assert after["compass_heading"] == pytest.approx(90.0, abs=2.0)
    # GPS moved east of the origin, not north
    assert after["position"]["longitude"] > -73.9846
    assert after["position"]["latitude"] == pytest.approx(40.7858, abs=1e-5)

    _run(boat.left.stop())
    _run(boat.right.stop())
    assert not _run(boat.left.is_moving())
    time.sleep(0.3)
    coasting = _run(boat.sensor.get_readings())
    assert coasting["speed_mps"] < after["speed_mps"]


def test_differential_thrust_turns_the_boat(boat):
    _park(boat)
    _run(boat.left.set_power(1.0))
    _run(boat.right.set_power(-1.0))
    time.sleep(0.8)
    readings = _run(boat.sensor.get_readings())
    _run(boat.left.stop())
    _run(boat.right.stop())
    # left forward + right reverse spins the nose right (clockwise), so the
    # compass heading increases from east
    assert 95.0 < readings["compass_heading"] < 270.0
    assert readings["angular_velocity"]["z"] < 0.0
    assert math.hypot(readings["sim_position_m"]["x"], readings["sim_position_m"]["y"]) < 0.5


def test_world_set_prop_pose_moves_the_floating_prop(boat):
    yaw = math.pi / 2
    _run(
        boat.world.do_command(
            {
                "command": "set_prop_pose",
                "prop": "boat",
                "position": [5.0, -3.0, 0.0],
                "orientation_wxyz": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
            }
        )
    )
    readings = _run(boat.sensor.get_readings())
    assert readings["sim_position_m"]["x"] == pytest.approx(5.0, abs=0.05)
    assert readings["sim_position_m"]["y"] == pytest.approx(-3.0, abs=0.05)
    assert readings["compass_heading"] == pytest.approx(0.0, abs=1.0)
    assert readings["speed_mps"] == pytest.approx(0.0, abs=1e-6)
    _park(boat)
    readings = _run(boat.sensor.get_readings())
    assert readings["sim_position_m"]["x"] == pytest.approx(0.0, abs=0.05)
    assert readings["compass_heading"] == pytest.approx(90.0, abs=1.0)


def test_movement_sensor_api_surface(boat):
    _park(boat)
    point, altitude = _run(boat.sensor.get_position())
    assert point.latitude == pytest.approx(40.7858, abs=1e-6)
    assert point.longitude == pytest.approx(-73.9846, abs=1e-6)
    assert altitude == pytest.approx(_spec().rest_z_m)
    heading = _run(boat.sensor.get_compass_heading())
    assert heading == pytest.approx(90.0, abs=1.0)
    orientation = _run(boat.sensor.get_orientation())
    assert orientation.o_z == pytest.approx(1.0)
    accel = _run(boat.sensor.get_linear_acceleration())
    assert accel.z == pytest.approx(GRAVITY_MPS2, abs=0.5)
    props = _run(boat.sensor.get_properties())
    assert props.position_supported and props.compass_heading_supported
    accuracy = _run(boat.sensor.get_accuracy())
    assert accuracy.position_nmea_gga_fix == 4
    velocity = _run(boat.sensor.get_linear_velocity())
    assert abs(velocity.x) < 0.05


def test_motor_go_for_integrates_revolutions_and_stops(boat):
    _park(boat)
    _run(boat.left.reset_zero_position(0.0))
    start = _run(boat.left.get_position())
    _run(boat.left.go_for(rpm=3000, revolutions=25))  # 0.5 s at max rpm
    assert not _run(boat.left.is_moving())
    revs = _run(boat.left.get_position())
    assert revs - start == pytest.approx(25.0, abs=4.0)
    props = _run(boat.left.get_properties())
    assert props.position_reporting is True
    status = _run(boat.left.do_command({"command": "status"}))
    assert status["body"] == "boat" and status["max_thrust_n"] == 49.0
    with pytest.raises(ValueError, match="rpm"):
        _run(boat.left.go_for(rpm=0, revolutions=1))


def test_motor_reconfigure_keeps_live_limits_and_odometry(boat):
    before = boat.left._h()
    boat.left.reconfigure(
        _config(
            "left-thruster",
            {"world": "sim-world", "prop": "boat", "position": [-0.45, 0.23, -0.1], "max_thrust_n": 30},
        ),
        {},
    )
    assert boat.left._h() is before
    assert before.max_thrust_n == 30.0
    # a moved thruster is re-registered on the body, keeping its odometry
    before.reset_revolutions(-7.0)
    boat.left.reconfigure(
        _config(
            "left-thruster",
            {"world": "sim-world", "prop": "boat", "position": [-0.5, 0.23, -0.1], "max_thrust_n": 49},
        ),
        {},
    )
    assert boat.left._h() is not before
    assert boat.left._h().revolutions() == pytest.approx(7.0)
    assert {t.name for t in boat.left._h().body.thrusters()} == {"left-thruster", "right-thruster"}


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------


def test_validation_requires_world_and_a_body():
    deps, _ = IsaacMotor.validate_config(_config("m", {"world": "sim-world", "prop": "boat"}))
    assert list(deps) == ["sim-world"]
    deps, _ = IsaacMovementSensor.validate_config(
        _config("s", {"world": "sim-world", "prim_path": "/World/thing"})
    )
    assert list(deps) == ["sim-world"]
    with pytest.raises(ValueError, match="prop"):
        IsaacMotor.validate_config(_config("m", {"world": "sim-world"}))
    with pytest.raises(ValueError, match="world"):
        IsaacMovementSensor.validate_config(_config("s", {"prop": "boat"}))


def test_motor_on_a_missing_or_fixed_prop_fails_clearly(world):
    with pytest.raises(ValueError, match="unknown prop"):
        IsaacMotor.new(_config("lonely", {"world": "sim-world", "prop": "ghost"}), {})
    with pytest.raises(ValueError, match="fixed"):
        IsaacMotor.new(_config("stuck", {"world": "sim-world", "prop": "piling"}), {})
    with pytest.raises(ValueError, match="mock mode"):
        IsaacMotor.new(_config("prim", {"world": "sim-world", "prim_path": "/World/x"}), {})


# ----------------------------------------------------------------------
# Isaac body handle against a fake RigidPrim view
# ----------------------------------------------------------------------


class FakeView:
    def __init__(self, position, orientation):
        import numpy as np

        self.position = np.array([position], dtype=np.float32)
        self.orientation = np.array([orientation], dtype=np.float32)
        self.velocity = np.zeros((1, 6), dtype=np.float32)
        self.applied = []

    def get_world_poses(self):
        return self.position.copy(), self.orientation.copy()

    def get_velocities(self):
        return self.velocity.copy()

    def apply_forces_and_torques_at_pos(self, forces=None, torques=None, positions=None, is_global=True):
        self.applied.append((forces.copy(), torques.copy(), is_global))


def test_isaac_body_handle_applies_one_wrench_per_step():
    spec = _spec()
    sim = SimpleNamespace(world=SimpleNamespace(current_time=3.0))
    view = FakeView((0.0, 0.0, spec.rest_z_m - 0.05), IDENTITY)
    handle = IsaacBodyHandle(sim, "boat", spec, (0.0, 0.0, spec.rest_z_m), IDENTITY, view)
    thruster = handle.register_thruster("t", (-0.4, 0.0, 0.0), (1.0, 0.0, 0.0))
    thruster.force_n = 10.0
    thruster.rpm = 600.0

    handle._on_physics_step(1.0 / 60.0)

    assert len(view.applied) == 1
    forces, torques, is_global = view.applied[0]
    assert is_global is True
    assert forces[0][0] == pytest.approx(10.0)
    assert forces[0][2] > spec.mass_kg * GRAVITY_MPS2  # sitting low: extra lift
    state = handle.get_state()
    assert state.sim_time_s == 3.0
    assert thruster.revolutions == pytest.approx(600.0 / 60.0 / 60.0)
    assert quat_to_rpy(state.orientation_wxyz) == pytest.approx((0.0, 0.0, 0.0))


def test_isaac_body_handle_skips_the_apply_when_nothing_acts():
    sim = SimpleNamespace(world=SimpleNamespace(current_time=0.0))
    view = FakeView((0.0, 0.0, 1.0), IDENTITY)
    handle = IsaacBodyHandle(sim, "crate", None, (0.0, 0.0, 1.0), IDENTITY, view)
    handle._on_physics_step(0.01)
    assert view.applied == []
    assert handle.get_state().position == pytest.approx((0.0, 0.0, 1.0))


def test_isaac_body_handle_survives_a_physics_error_once():
    sim = SimpleNamespace(world=SimpleNamespace(current_time=0.0))

    class BrokenView(FakeView):
        def get_velocities(self):
            raise RuntimeError("physics view not ready")

    view = BrokenView((0.0, 0.0, 0.0), IDENTITY)
    handle = IsaacBodyHandle(sim, "boat", _spec(), (0.0, 0.0, 0.0), IDENTITY, view)
    handle._on_physics_step(0.01)
    handle._on_physics_step(0.01)
    assert view.applied == []
    assert handle._warned is True


# ----------------------------------------------------------------------
# review follow-ups
# ----------------------------------------------------------------------


def test_mock_get_prop_pose_follows_the_moving_body(boat):
    """get_prop_pose reports where the body is now, not where it spawned."""
    _park(boat)
    start = _run(boat.world.do_command({"command": "get_prop_pose", "prop": "boat"}))
    _run(boat.left.set_power(1.0))
    _run(boat.right.set_power(1.0))
    time.sleep(0.6)
    moved = _run(boat.world.do_command({"command": "get_prop_pose", "prop": "boat"}))
    _run(boat.left.stop())
    _run(boat.right.stop())
    assert moved["pose"]["position"][0] > start["pose"]["position"][0] + 0.2
    readings = _run(boat.sensor.get_readings())
    assert moved["pose"]["position"][0] == pytest.approx(readings["sim_position_m"]["x"], abs=0.1)


def test_mock_set_prop_pose_returns_the_floated_z(boat):
    """A body snaps to its rest draft; the reported pose says so."""
    result = _run(
        boat.world.do_command(
            {
                "command": "set_prop_pose",
                "prop": "boat",
                "position": [1.0, 1.0, 5.0],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        )
    )
    assert result["position"][2] == pytest.approx(_spec().rest_z_m)
    queried = _run(boat.world.do_command({"command": "get_prop_pose", "prop": "boat"}))
    assert queried["pose"]["position"][2] == pytest.approx(_spec().rest_z_m)
    _park(boat)


def test_teleport_does_not_spike_the_imu(boat):
    _park(boat)
    _run(boat.left.set_power(1.0))
    _run(boat.right.set_power(1.0))
    time.sleep(0.5)
    _run(boat.left.stop())
    _run(boat.right.stop())
    # teleport while still moving fast, then read the very next samples
    _run(boat.world.do_command({"command": "park_props", "props": ["boat"]}))
    worst = 0.0
    for _ in range(5):
        accel = _run(boat.sensor.get_linear_acceleration())
        worst = max(worst, abs(accel.x), abs(accel.y))
        time.sleep(0.02)
    assert worst < 5.0  # a 1.5 m/s jump over one 10 ms step would read 150+


def test_removed_motor_detaches_from_the_body(world):
    extra = IsaacMotor.new(
        _config("extra-thruster", {"world": "sim-world", "prop": "boat", "position": [0, 0, 0]}),
        {},
    )
    body = extra._h().body
    _run(extra.set_power(0.5))
    assert "extra-thruster" in {t.name for t in body.thrusters()}
    _run(extra.close())
    assert "extra-thruster" not in {t.name for t in body.thrusters()}
    assert "extra-thruster" not in SimManager.get()._handles
    with pytest.raises(RuntimeError, match="not attached"):
        _run(extra.set_power(0.5))


def test_go_for_waits_on_sim_time_odometry(boat, monkeypatch):
    _park(boat)
    _run(boat.left.reset_zero_position(0.0))
    handle = boat.left._h()
    # freeze the clock the odometry integrates against: a wall-clock sleep
    # would return long before the revolutions arrive
    original_tick = handle._state.tick
    ticks = []

    def slow_tick(dt):
        ticks.append(dt)
        original_tick(dt * 0.25)

    monkeypatch.setattr(handle._state, "tick", slow_tick)
    started = time.monotonic()
    _run(boat.left.go_for(rpm=3000, revolutions=5))  # 0.1 s of sim time at max rpm
    elapsed = time.monotonic() - started
    assert elapsed > 0.3  # four times slower than the wall clock
    assert _run(boat.left.get_position()) == pytest.approx(5.0, abs=1.0)
    assert not _run(boat.left.is_moving())


def test_go_for_times_out_when_the_sim_is_paused(boat, monkeypatch):
    import isaac_module.models.motor as motor_module

    _park(boat)
    handle = boat.left._h()
    monkeypatch.setattr(handle._state, "tick", lambda dt: None)  # odometry frozen
    monkeypatch.setattr(SimManager.get(), "sim_time_s", lambda: 1.0)  # clock frozen
    monkeypatch.setattr(motor_module, "GO_FOR_STALL_TIMEOUT_S", 0.1)
    with pytest.raises(TimeoutError, match="stopped advancing"):
        _run(boat.left.go_for(rpm=3000, revolutions=5))
    assert not _run(boat.left.is_moving())


def test_prim_path_bodies_are_keyed_by_full_path():
    from isaac_module.sim_manager import _body_key

    assert _body_key("/World/A/hull") == "World_A_hull"
    assert _body_key("/World/B/hull") == "World_B_hull"
    assert _body_key("/World/A/hull") != _body_key("/World/B/hull")
