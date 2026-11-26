"""Servo setpoints (teleop) and joint-position snapshots on IsaacArmHandle:
neither waits for the sim thread once attached, and the arm interpolates
between setpoints at physics rate instead of jumping to each one."""

import math

import pytest
from test_arm_is_moving import _FakeArt, _FakeSim

import isaac_module.sim_manager as sm
from isaac_module.sim_manager import IsaacArmHandle, JointTrajectoryPoint


class _CountingSim(_FakeSim):
    def __init__(self):
        super().__init__()
        self.runs = 0

    def run(self, fn, timeout=None):
        self.runs += 1
        return fn()


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(sm.time, "monotonic", c)
    return c


def _handle(n=2, **kw):
    sim, art = _CountingSim(), _FakeArt([0.0] * n)
    return sim, art, IsaacArmHandle(sim, art, None, "/World/arm", **kw)


def _servo_step(sim, h):
    return sim.world.callbacks[h._servo_callback_name]


def test_first_setpoint_attaches_then_later_ones_skip_the_sim_thread(clock):
    sim, art, h = _handle()
    h.servo_target([0.1, 0.1])
    assert h._servo_callback_name in sim.world.callbacks
    runs = sim.runs
    for k in range(5):
        clock.now += 0.05
        h.servo_target([0.1 + k * 0.01, 0.1])
    assert sim.runs == runs


def test_interpolates_to_setpoint_over_the_period(clock):
    sim, art, h = _handle()
    h.set_joint_targets([0.0, 0.0])
    h.servo_target([0.2, -0.2])
    step = _servo_step(sim, h)
    period = h._servo_period_s
    step(period / 4)
    assert art.actions[-1].joint_positions.tolist() == pytest.approx([0.05, -0.05])
    step(period / 4)
    assert h._commanded == pytest.approx([0.1, -0.1])
    step(period)  # past the period: lands exactly on the setpoint
    assert h._commanded == pytest.approx([0.2, -0.2])
    assert h._servo_goal is None
    actions = len(art.actions)
    step(period)
    assert len(art.actions) == actions  # idle once arrived


def test_new_setpoint_continues_from_the_commanded_pose(clock):
    sim, art, h = _handle()
    h.set_joint_targets([0.0, 0.0])
    h.servo_target([0.2, 0.0])
    step = _servo_step(sim, h)
    step(h._servo_period_s / 2)
    assert h._commanded == pytest.approx([0.1, 0.0])
    clock.now += 0.05
    h.servo_target([0.1, 0.4])
    step(h._servo_period_s / 2)
    # halfway from (0.1, 0) to (0.1, 0.4): no jump back toward the old start
    assert h._commanded == pytest.approx([0.1, 0.2])


def test_period_tracks_setpoint_interval_and_ignores_long_gaps(clock):
    sim, art, h = _handle()
    for _ in range(40):
        h.servo_target([0.0, 0.0])
        clock.now += 0.07
    assert h._servo_period_s == pytest.approx(0.07, rel=0.02)
    clock.now += 5.0  # the operator paused: not a setpoint interval
    h.servo_target([0.0, 0.0])
    assert h._servo_period_s == pytest.approx(0.07, rel=0.02)
    for _ in range(40):
        clock.now += 1.0 / 30
        h.servo_target([0.0, 0.0])
    assert h._servo_period_s == pytest.approx(1.0 / 30, rel=0.02)


def test_period_is_clamped(clock):
    sim, art, h = _handle()
    for _ in range(40):
        h.servo_target([0.0, 0.0])
        clock.now += 0.001
    assert h._servo_period_s == sm.SERVO_MIN_PERIOD_S
    for _ in range(40):
        h.servo_target([0.0, 0.0])
        clock.now += 0.29
    assert h._servo_period_s == sm.SERVO_MAX_PERIOD_S


def test_velocity_limit_caps_each_step(clock):
    sim, art, h = _handle(max_joint_deg_s=30.0)
    h.set_joint_targets([0.0, 0.0])
    h.servo_target([1.0, 0.0])
    _servo_step(sim, h)(1.0)  # interpolation says arrive; the cap says 30 deg
    assert h._commanded == pytest.approx([math.radians(30.0), 0.0])
    assert h._servo_goal == [1.0, 0.0]  # keeps going next step


def test_blocking_targets_and_stop_cancel_servo_motion(clock):
    sim, art, h = _handle()
    h.set_joint_targets([0.0, 0.0])
    h.servo_target([0.4, 0.4])
    step = _servo_step(sim, h)
    step(h._servo_period_s / 4)
    h.set_joint_targets([-0.3, 0.0])
    step(h._servo_period_s)
    assert h._commanded == [-0.3, 0.0]

    h.servo_target([0.4, 0.4])
    h.stop()
    held = list(h._commanded)
    step(h._servo_period_s)
    assert h._commanded == held


def test_servo_waits_while_a_timestamped_trajectory_runs(clock):
    sim, art, h = _handle()
    h.servo_target([0.0, 0.0])
    h.start_trajectory(
        (JointTrajectoryPoint(0.0, (0.1, 0.1)), JointTrajectoryPoint(1.0, (0.2, 0.2)))
    )
    h.servo_target([0.9, 0.9])
    _servo_step(sim, h)(0.01)
    assert h._servo_goal is None


def test_reads_use_the_physics_step_snapshot(clock):
    sim, art, h = _handle()
    assert h.get_joint_positions() == [0.0, 0.0]  # first read attaches
    runs = sim.runs
    art.positions = [0.3, -0.3]
    _servo_step(sim, h)(1 / 60)
    assert h.get_joint_positions() == pytest.approx([0.3, -0.3])
    assert sim.runs == runs

    clock.now += sm.JOINT_SNAPSHOT_MAX_AGE_S + 0.01  # sim paused: no steps
    art.positions = [0.5, 0.5]
    assert h.get_joint_positions() == pytest.approx([0.5, 0.5])
    assert sim.runs == runs + 1


def test_world_reset_drops_servo_state_and_snapshot(clock):
    sim, art, h = _handle()
    h.servo_target([0.4, 0.4])
    _servo_step(sim, h)(0.0)

    class World:
        def reset(self):
            pass

    mgr = sm.SimManager()
    mgr.world = World()
    mgr._handles["arm"] = ({}, h)
    mgr._reset_world()
    assert h._servo_goal is None and h._servo_inbox is None and h._snapshot is None
