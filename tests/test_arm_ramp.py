import pytest
from test_arm_is_moving import _FakeArt, _FakeSim

from isaac_module.sim_manager import IsaacArmHandle, JointTrajectoryPoint, ramp_toward


def test_clamps_each_joint_independently():
    assert ramp_toward([0.0, 0.0], [1.0, 0.05], 0.1) == pytest.approx([0.1, 0.05])


def test_lands_exactly_on_goal_without_overshoot():
    goal = [0.3000000001, -0.7]
    out = ramp_toward([0.25, -0.65], goal, 0.1)
    assert out == goal  # exact, so the callback can tell the ramp is done


def test_negative_direction():
    assert ramp_toward([0.0], [-1.0], 0.25) == pytest.approx([-0.25])


def test_zero_step_leaves_values():
    assert ramp_toward([0.2, 0.4], [1.0, 1.0], 0.0) == pytest.approx([0.2, 0.4])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        ramp_toward([0.0], [0.0, 1.0], 0.1)


def _handle(n=2, **kw):
    sim, art = _FakeSim(), _FakeArt([0.0] * n)
    return sim, art, IsaacArmHandle(sim, art, None, "/World/arm", **kw)


def test_unset_limit_applies_goal_immediately_and_tracks_commanded():
    sim, art, h = _handle()
    h.set_joint_targets([0.5, -0.5])
    assert art.actions[-1].joint_positions.tolist() == pytest.approx([0.5, -0.5])
    assert h._target == h._commanded == [0.5, -0.5]
    assert sim.world.callbacks == {}


def test_trajectory_step_sets_commanded():
    sim, art, h = _handle()
    h.start_trajectory((JointTrajectoryPoint(0.0, (0.1, 0.2)),))
    next(iter(sim.world.callbacks.values()))(0.01)
    assert h._commanded == [0.1, 0.2]


def test_stop_holds_measured_minus_nudge():
    sim, art, h = _handle()
    h.set_joint_targets([0.4, 0.4])
    h.set_nudge([0.1, 0.0])  # measured becomes 0.5, 0.4
    h.stop()
    assert h._target == pytest.approx([0.4, 0.4])
    assert h._commanded == pytest.approx([0.4, 0.4])
