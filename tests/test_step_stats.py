import pytest

from isaac_module.sim_manager import StepStats


def test_empty_window_reports_zeros():
    assert StepStats(clock=lambda: 0.0).snapshot() == {
        "real_time_factor": 0.0,
        "step_ms": 0.0,
        "loop_ms": 0.0,
        "render_hz": 0.0,
    }


def test_window_averages_and_rates():
    now = [0.0]
    stats = StepStats(window_s=2.0, clock=lambda: now[0])
    for _ in range(4):
        now[0] += 0.025
        # 20 ms in world.step, 25 ms per loop, 1/30 s of sim per loop
        stats.record(step_s=0.020, loop_s=0.025, sim_dt=1 / 30)
    snap = stats.snapshot()
    assert snap["step_ms"] == pytest.approx(20.0)
    assert snap["loop_ms"] == pytest.approx(25.0)
    assert snap["render_hz"] == pytest.approx(40.0)
    assert snap["real_time_factor"] == pytest.approx((1 / 30) / 0.025)


def test_window_drops_old_samples():
    now = [0.0]
    stats = StepStats(window_s=2.0, clock=lambda: now[0])
    stats.record(step_s=1.0, loop_s=1.0, sim_dt=0.0)
    now[0] = 3.0
    stats.record(step_s=0.010, loop_s=0.010, sim_dt=0.010)
    assert stats.snapshot()["step_ms"] == pytest.approx(10.0)
