import itertools
import math

import pytest
from test_arm_is_moving import _FakeArt, _FakeSim

from isaac_module.sim_manager import (
    IsaacArmHandle,
    JointTrajectoryPoint,
    SimManager,
    profile_step,
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


def test_profile_accelerates_cruises_and_stops_at_the_end():
    remaining, speed, dt = 1.0, 0.0, 0.01
    speeds = []
    for _ in range(1000):
        advance, speed = profile_step(remaining, speed, 1.0, 4.0, dt)
        remaining -= advance
        speeds.append(speed)
        if remaining <= 0.0:
            break
    assert remaining == pytest.approx(0.0, abs=1e-12) and speeds[-1] == 0.0
    assert speeds[0] == pytest.approx(0.04)  # a * dt
    assert max(speeds) == pytest.approx(1.0)  # reaches the speed limit
    # never speeds up faster than the limit; slows before the end
    assert all(b - a <= 4.0 * dt + 1e-9 for a, b in itertools.pairwise(speeds))
    assert speeds[-2] < 0.5
    # trapezoid: 0.25 s up, 0.75 s cruise, 0.25 s down
    assert len(speeds) * dt == pytest.approx(1.25, abs=0.05)


def test_profile_without_acceleration_limit_runs_at_full_speed():
    assert profile_step(1.0, 0.0, 0.5, None, 0.1) == pytest.approx((0.05, 0.5))
    assert profile_step(0.01, 0.5, 0.5, None, 0.1) == (0.01, 0.0)
    assert profile_step(0.0, 0.5, 0.5, 2.0, 0.1) == (0.0, 0.0)


def _run(sim, h, steps, dt=1 / 60):
    ramp = sim.world.callbacks[h._ramp_callback_name]
    out = []
    for _ in range(steps):
        ramp(dt)
        out.append(list(h._commanded))
    return out


def test_move_ramps_up_and_down_under_the_acceleration_limit():
    sim, art, h = _handle(max_joint_deg_s=180.0, max_joint_deg_s2=720.0)
    h.set_joint_targets([0.0, 0.0])
    h.set_joint_targets([math.radians(90.0), 0.0])
    trace = [p[0] for p in _run(sim, h, 120)]
    assert trace[-1] == math.radians(90.0)  # lands exactly
    steps = [b - a for a, b in zip([0.0, *trace], trace)]
    assert steps[0] == pytest.approx(math.radians(720.0) / 60 / 60)  # a * dt * dt
    assert max(steps) == pytest.approx(math.radians(180.0) / 60)
    moving = [d for d in steps if d > 0]
    assert moving[-1] < max(steps) / 2  # slows into the goal
    # 90 deg at 180 deg/s with 0.25 s ramps takes 0.75 s
    assert len(moving) / 60 == pytest.approx(0.75, abs=0.05)
    assert not h.path_active()


def test_path_passes_through_waypoints_without_stopping():
    sim, art, h = _handle(max_joint_deg_s=60.0, max_joint_deg_s2=600.0)
    h.set_joint_targets([0.0, 0.0])
    h.follow_path([[0.5, 0.0], [0.5, 0.5]])
    assert h.path_active()
    trace = _run(sim, h, 200)
    assert trace[-1] == [0.5, 0.5]
    corner = next(i for i, p in enumerate(trace) if p[1] > 0.0)
    assert trace[corner - 1][0] == pytest.approx(0.5, abs=math.radians(60.0) / 60)
    assert all(p[0] == 0.5 for p in trace[corner:])  # second leg stays on its segment
    # still at cruise speed across the corner: no stop at the waypoint
    travelled = sum(abs(b - a) for a, b in zip(trace[corner - 2], trace[corner - 1]))
    assert travelled == pytest.approx(math.radians(60.0) / 60)


def test_retarget_mid_move_keeps_speed_forward_and_drops_it_in_reverse():
    sim, art, h = _handle(max_joint_deg_s=60.0, max_joint_deg_s2=120.0)
    h.set_joint_targets([0.0, 0.0])
    h.set_joint_targets([1.0, 0.0])
    _run(sim, h, 40)
    cruising = h._path_speed
    assert cruising == pytest.approx(math.radians(60.0))
    h.set_joint_targets([2.0, 0.0])
    assert h._path_speed == pytest.approx(cruising)
    h.set_joint_targets([-1.0, 0.0])
    assert h._path_speed == 0.0
