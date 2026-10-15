"""Arm movement reporting at the component and simulator-handle layers."""

import asyncio
import math
import threading
from concurrent.futures import CancelledError, Future
from datetime import timedelta

import pytest
from viam.components.arm import Arm, JointPositions
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import (
    ArmHandle,
    IsaacArmHandle,
    JointTrajectoryPoint,
    SimManager,
    UR_JOINT_NAMES,
)


class _FakeArt:
    def __init__(self, vels):
        self._vels = vels
        self.dof_names = list(range(len(vels)))
        self._articulation_view = self
        self.max_velocity_calls = []
        self.positions = [0.0] * len(vels)
        self.actions = []

    def get_joint_velocities(self, joint_indices=None):
        if joint_indices is None:
            return self._vels
        return [self._vels[index] for index in joint_indices]

    def set_joint_velocities(self, values, *, joint_indices):
        for value, index in zip(values, joint_indices):
            self._vels[index] = float(value)

    def set_max_joint_velocities(self, values, *, joint_indices):
        self.max_velocity_calls.append((values, joint_indices))

    def get_joint_max_velocities(self):
        values = [float("inf")] * len(self.dof_names)
        if self.max_velocity_calls:
            configured, indices = self.max_velocity_calls[-1]
            for value, index in zip(configured[0], indices):
                values[index] = value
        return [values]

    def get_joint_positions(self, joint_indices=None):
        if joint_indices is None:
            return self.positions
        return [self.positions[index] for index in joint_indices]

    def apply_action(self, action):
        self.actions.append(action)
        for value, index in zip(action.joint_positions, action.joint_indices):
            self.positions[index] = float(value)


class _FakeWorld:
    def __init__(self):
        self.callbacks = {}

    def add_physics_callback(self, name, callback):
        self.callbacks[name] = callback

    def remove_physics_callback(self, name):
        del self.callbacks[name]


class _FakeAction:
    def __init__(self, *, joint_positions, joint_indices):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class _FakeIsaac:
    ArticulationAction = _FakeAction


class _FakeSim:
    def __init__(self):
        self.world = _FakeWorld()
        self._step_callbacks = {}
        self._isaac = _FakeIsaac()

    def run(self, fn, timeout=None):
        return fn()


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture(scope="module")
def sim():
    manager = SimManager.get()
    t = threading.Thread(target=manager.main_loop, daemon=True)
    t.start()
    yield manager
    manager.request_stop()
    t.join(timeout=5)


def test_settled_arm_is_not_moving():
    # residual velocities Isaac's PD drives produce while holding pose
    handle = IsaacArmHandle(
        _FakeSim(), _FakeArt([5e-3, -2e-3, 1e-3, 0, 0, 0]), None, "/World/arm"
    )
    assert handle.is_moving() is False


def test_movement_velocity_threshold_is_configurable():
    velocities = [0.011, 0, 0, 0, 0, 0]
    default_handle = IsaacArmHandle(
        _FakeSim(), _FakeArt(velocities), None, "/World/arm"
    )
    tolerant_handle = IsaacArmHandle(
        _FakeSim(),
        _FakeArt(velocities),
        None,
        "/World/arm",
        is_moving_velocity_threshold_rad_s=0.02,
    )

    assert default_handle.is_moving() is True
    assert tolerant_handle.is_moving() is False


@pytest.mark.parametrize("value", (True, "0.02", -0.01, float("inf")))
def test_movement_velocity_threshold_must_be_non_negative_finite_number(value):
    config = ComponentConfig(
        name="arm",
        attributes=dict_to_struct(
            {
                "world": "sim-world",
                "asset": "lite6",
                "is_moving_velocity_threshold_rad_s": value,
            }
        ),
    )

    with pytest.raises(ValueError, match="is_moving_velocity_threshold_rad_s"):
        IsaacArm.validate_config(config)


@pytest.mark.parametrize("value", (True, "180", 0, -1, float("inf"), float("nan")))
def test_max_joint_velocity_must_be_positive_finite_number(value):
    config = ComponentConfig(
        name="arm",
        attributes=dict_to_struct(
            {
                "world": "sim-world",
                "asset": "lite6",
                "max_joint_deg_s": value,
            }
        ),
    )

    with pytest.raises(ValueError, match="max_joint_deg_s"):
        IsaacArm.validate_config(config)


@pytest.mark.parametrize("value", (True, "60", 0, -1, float("inf"), float("nan")))
def test_max_joint_acceleration_must_be_positive_finite_number(value):
    config = ComponentConfig(
        name="arm",
        attributes=dict_to_struct(
            {
                "world": "sim-world",
                "asset": "lite6",
                "max_joint_deg_s2": value,
            }
        ),
    )

    with pytest.raises(ValueError, match="max_joint_deg_s2"):
        IsaacArm.validate_config(config)


def test_max_joint_velocity_is_applied_to_only_arm_dofs():
    art = _FakeArt([0.0] * 7)
    art.dof_names = list(UR_JOINT_NAMES) + ["finger_joint"]

    handle = IsaacArmHandle(
        _FakeSim(),
        art,
        None,
        "/World/arm",
        UR_JOINT_NAMES,
        max_joint_deg_s=180.0,
        max_joint_deg_s2=60.0,
    )

    assert len(art.max_velocity_calls) == 1
    values, joint_indices = art.max_velocity_calls[0]
    assert values.shape == (1, 6)
    assert values[0].tolist() == pytest.approx([math.pi] * 6)
    assert joint_indices.tolist() == list(range(6))
    metadata = handle.dynamics_metadata()
    assert metadata["max_velocity_rad_s"] == pytest.approx([math.pi] * 6)
    assert metadata["max_acceleration_rad_s2"] == pytest.approx(
        [math.radians(60.0)] * 6
    )


def test_actively_moving_arm_reports_moving():
    handle = IsaacArmHandle(
        _FakeSim(), _FakeArt([0.2, 0, 0, 0, 0, 0]), None, "/World/arm"
    )
    assert handle.is_moving() is True


def test_gripper_velocity_does_not_make_arm_report_moving():
    art = _FakeArt([0.0] * 6 + [0.5])
    art.dof_names = list(UR_JOINT_NAMES) + ["finger_joint"]
    handle = IsaacArmHandle(
        _FakeSim(), art, None, "/World/arm", UR_JOINT_NAMES
    )
    assert handle.is_moving() is False


def test_timestamped_trajectory_advances_from_physics_steps():
    sim = _FakeSim()
    art = _FakeArt([0.0, 0.0])
    handle = IsaacArmHandle(sim, art, None, "/World/arm")
    future = handle.start_trajectory(
        (
            JointTrajectoryPoint(0.0, (0.1, 0.2)),
            JointTrajectoryPoint(0.1, (0.3, 0.4)),
        )
    )
    callback = next(iter(sim.world.callbacks.values()))

    callback(0.05)
    assert [action.joint_positions.tolist() for action in art.actions] == [
        pytest.approx([0.1, 0.2])
    ]
    assert not future.done()
    callback(0.05)
    assert not future.done()
    callback(0.05)
    assert future.result() is None
    assert art.actions[-1].joint_positions.tolist() == pytest.approx([0.3, 0.4])

    handle.stop_trajectory()
    assert sim.world.callbacks == {}
    assert art.actions[-1].joint_positions.tolist() == pytest.approx([0.3, 0.4])


def test_stopping_trajectory_cancels_callback_and_stops_at_measured_position():
    sim = _FakeSim()
    art = _FakeArt([0.0, 0.0])
    handle = IsaacArmHandle(sim, art, None, "/World/arm")
    future = handle.start_trajectory(
        (
            JointTrajectoryPoint(0.0, (0.1, 0.2)),
            JointTrajectoryPoint(1.0, (0.3, 0.4)),
        )
    )
    callback = next(iter(sim.world.callbacks.values()))
    callback(0.05)
    art.positions = [0.12, 0.18]
    art._vels = [0.11, -0.03]
    assert handle.is_moving() is True

    handle.stop_trajectory()

    assert sim.world.callbacks == {}
    with pytest.raises(CancelledError):
        future.result()
    assert art.positions == pytest.approx([0.12, 0.18])
    assert handle.is_moving() is False

    handle.stop_trajectory()
    assert art.positions == pytest.approx([0.12, 0.18])
    assert handle.is_moving() is False


def test_streamed_request_cancellation_stops_active_trajectory():
    class Handle:
        def __init__(self):
            self.future = Future()
            self.started = threading.Event()
            self.stop_calls = 0

        def start_trajectory(self, points):
            self.points = points
            self.started.set()
            return self.future

        def stop_trajectory(self):
            self.stop_calls += 1
            self.future.cancel()

    async def scenario():
        handle = Handle()
        arm = IsaacArm("arm")
        arm._handle = handle

        async def batches():
            yield [Arm.TrajectoryPoint(timedelta(0), [10.0, 20.0])]

        updates = arm.move_through_joint_positions_streamed(batches())
        request = asyncio.create_task(anext(updates))
        assert await asyncio.to_thread(handle.started.wait, 1.0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert handle.stop_calls == 1
        assert handle.points == [
            JointTrajectoryPoint(0.0, (math.radians(10.0), math.radians(20.0)))
        ]

    asyncio.run(scenario())


def test_rejected_stream_does_not_stop_the_active_trajectory():
    class Handle:
        def __init__(self):
            self.stop_calls = 0

        def start_trajectory(self, points):
            raise RuntimeError("another arm trajectory is already active")

        def stop_trajectory(self):
            self.stop_calls += 1

    async def scenario():
        handle = Handle()
        arm = IsaacArm("arm")
        arm._handle = handle

        async def batches():
            yield [Arm.TrajectoryPoint(timedelta(0), [10.0, 20.0])]

        updates = arm.move_through_joint_positions_streamed(batches())
        with pytest.raises(RuntimeError, match="already active"):
            await anext(updates)
        assert handle.stop_calls == 0

    asyncio.run(scenario())


def test_move_completes_at_target_despite_residual_velocity():
    class Handle(ArmHandle):
        def __init__(self):
            self._sim = _FakeSim()
            self.target = []

        def set_joint_targets(self, positions):
            self.target = positions

        def get_joint_positions(self):
            return self.target

        def _positions_on_sim_thread(self):
            return self.target

        def is_moving(self):
            return True

    arm = IsaacArm("arm")
    handle = Handle()
    arm._handle = handle
    asyncio.run(arm.move_to_joint_positions(JointPositions(values=[10.0, 20.0])))
    # the convergence check resolved on the first evaluation and removed itself
    assert handle._sim._step_callbacks == {}


def test_joint_target_wait_resolves_from_sim_steps_and_cancels_cleanly():
    sim = _FakeSim()
    art = _FakeArt([0.0, 0.0])
    handle = IsaacArmHandle(sim, art, None, "/World/arm")

    future = handle.wait_for_joint_targets([0.5, -0.5], math.radians(0.5))
    assert not future.done()
    (step,) = sim._step_callbacks.values()

    art.positions = [0.3, -0.3]
    step(1 / 60)
    assert not future.done()
    art.positions = [0.499, -0.501]
    step(1 / 60)
    assert future.result() is None
    assert sim._step_callbacks == {}

    stalled = handle.wait_for_joint_targets([1.0, 1.0], math.radians(0.5))
    (step,) = sim._step_callbacks.values()
    stalled.cancel()
    step(1 / 60)
    assert sim._step_callbacks == {}


def test_move_times_out_when_joints_never_arrive():
    class Handle(ArmHandle):
        def __init__(self):
            self._sim = _FakeSim()

        def set_joint_targets(self, positions):
            pass

        def get_joint_positions(self):
            return [0.0, 0.0]

        def _positions_on_sim_thread(self):
            return [0.0, 0.0]

    arm = IsaacArm("arm")
    arm._move_timeout = 0.05
    handle = Handle()
    arm._handle = handle
    with pytest.raises(TimeoutError, match="stuck joints: j0"):
        asyncio.run(arm.move_to_joint_positions(JointPositions(values=[10.0, 20.0])))
    assert handle._sim._step_callbacks == {} or all(
        callback(0.0) is None for callback in list(handle._sim._step_callbacks.values())
    )
    assert handle._sim._step_callbacks == {}
@pytest.fixture(scope="module")
def world(sim):
    return IsaacWorld.new(_config("sim-world", {"mock": True}), {})


def _arm(world):
    return IsaacArm.new(
        _config("my-arm", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {}
    )


def test_reports_not_moving_at_rest(world):
    arm = _arm(world)
    assert asyncio.run(arm.is_moving()) is False


def test_reports_moving_only_while_move_is_in_flight(world):
    arm = _arm(world)

    async def scenario():
        assert not await arm.is_moving()

        # MockArmHandle moves at SPEED=1 rad/s. A ~2 rad target takes ~2s — ample
        # time to sample mid-flight without racing convergence.
        target = JointPositions(values=[math.degrees(2.0)] + [0.0] * 5)
        move_task = asyncio.create_task(arm.move_to_joint_positions(target))
        await asyncio.sleep(0.2)
        assert await arm.is_moving()

        await move_task
        assert not await arm.is_moving()

    asyncio.run(scenario())


def test_reports_moving_across_overlapping_moves(world):
    """is_moving stays True from the first move's start until the last move ends."""
    arm = _arm(world)

    async def scenario():
        first = JointPositions(values=[math.degrees(1.5)] + [0.0] * 5)
        second = JointPositions(values=[math.degrees(0.5)] + [0.0] * 5)

        task_a = asyncio.create_task(arm.move_to_joint_positions(first))
        await asyncio.sleep(0.1)
        task_b = asyncio.create_task(arm.move_to_joint_positions(second))
        await asyncio.sleep(0.1)
        assert await arm.is_moving()

        await task_a
        assert await arm.is_moving()

        await task_b
        assert not await arm.is_moving()

    asyncio.run(scenario())
