"""IsaacArm.is_moving is True iff a MoveTo* RPC is in flight."""

import asyncio
import math

import pytest
from viam.components.arm import JointPositions
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.world import IsaacWorld


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture(scope="module")
def world():
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
