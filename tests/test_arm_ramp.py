import pytest
from isaac_module.sim_manager import ramp_toward


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
