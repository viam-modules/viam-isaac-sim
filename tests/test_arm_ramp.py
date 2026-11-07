import math

import pytest
from test_arm_is_moving import _FakeArt, _FakeSim

from isaac_module.sim_manager import (
    IsaacArmHandle,
    JointTrajectoryPoint,
    SimManager,
    ramp_toward,
)


def test_scales_all_joints_along_the_segment():
    assert ramp_toward([0.0, 0.0], [1.0, 0.5], 0.1) == pytest.approx([0.1, 0.05])


def test_lands_exactly_on_goal_without_overshoot():
    goal = [0.3000000001, -0.7]
    out = ramp_toward([0.25, -0.65], goal, 0.1)
    assert out == goal  # exact, so the callback can tell the ramp is done


def test_negative_direction():
    assert ramp_toward([0.0], [-1.0], 0.25) == pytest.approx([-0.25])


def test_zero_step_leaves_values():
    assert ramp_toward([0.2, 0.4], [1.0, 1.0], 0.0) == pytest.approx([0.2, 0.4])


def test_nonpositive_step_returns_current():
    assert ramp_toward([0.2, 0.4], [1.0, 1.0], -1.0) == [0.2, 0.4]


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


def test_limit_ramps_target_by_v_dt_per_step():
    sim, art, h = _handle(max_joint_deg_s=60.0)  # 60 deg/s
    name = f"arm_ramp_{id(h)}"
    assert name in sim.world.callbacks
    h.set_joint_targets([math.radians(30), 0.0])
    assert art.actions == []  # goal not applied immediately
    step = sim.world.callbacks[name]
    step(0.1)  # 6 deg
    assert art.actions[-1].joint_positions.tolist() == pytest.approx(
        [math.radians(6), 0.0]
    )
    for _ in range(10):
        step(0.1)
    assert h._commanded == h._target
    applied = len(art.actions)
    step(0.1)
    assert len(art.actions) == applied  # idle once arrived


def test_latch_after_trajectory_ramps_from_trajectory_pose():
    sim, art, h = _handle(max_joint_deg_s=60.0)
    h.start_trajectory((JointTrajectoryPoint(0.0, (0.5, 0.0)),))
    traj = next(
        cb for n, cb in sim.world.callbacks.items() if n.startswith("arm_trajectory")
    )
    traj(0.01)
    h.set_joint_targets([0.0, 0.0])
    sim.world.callbacks[f"arm_ramp_{id(h)}"](0.1)
    assert h._commanded == pytest.approx([0.5 - math.radians(6), 0.0])


def test_nudge_rides_on_ramped_target():
    sim, art, h = _handle(max_joint_deg_s=60.0)
    h.set_joint_targets([1.0, 0.0])
    h.set_nudge([0.0, 0.2])
    sim.world.callbacks[f"arm_ramp_{id(h)}"](0.1)
    assert art.actions[-1].joint_positions.tolist() == pytest.approx(
        [math.radians(6), 0.2]
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_set_joint_targets_rejects_non_finite(bad):
    sim, art, h = _handle(max_joint_deg_s=60.0)
    with pytest.raises(ValueError, match="finite"):
        h.set_joint_targets([0.0, bad])
    assert h._target is None


def test_reset_world_clears_ramp_state_before_resetting():
    mgr = SimManager()
    seen = []
    sim, art, h = _handle(max_joint_deg_s=60.0)
    h.set_joint_targets([1.0, 1.0])

    class World:
        def reset(self):
            seen.append((h._target, h._commanded))

    mgr.world = World()
    mgr._handles["arm"] = ({}, h)
    mgr._reset_world()
    assert seen == [(None, None)]  # cleared before world.reset ran
