"""Moves honor the `extra` keys the RDK teleop pipeline sends:
{"waitAtEnd": false, "interpolate": false} (viam-ufactory-xarm semantics)."""

import asyncio
import math
from concurrent.futures import Future

import pytest
from viam.components.arm import JointPositions

from isaac_module.models.arm import IsaacArm

TELEOP = {"waitAtEnd": False, "interpolate": False}


class _Handle:
    """Records targets; joints never move, so any convergence wait would time out."""

    def __init__(self):
        self.targets = []
        self.servo = []
        self.stops = 0

    def set_joint_targets(self, positions):
        self.targets.append(list(positions))

    def servo_target(self, positions):
        self.servo.append(list(positions))

    def get_joint_positions(self):
        return [0.0, 0.0]

    def wait_for_joint_targets(self, targets, tolerance):
        return Future()  # never resolves

    def stop(self):
        self.stops += 1


def _arm():
    arm = IsaacArm("arm")
    arm._handle = _Handle()
    arm._move_timeout = 0.05
    return arm


def _jp(*deg):
    return JointPositions(values=list(deg))


def test_teleop_move_through_servos_to_last_waypoint_and_returns():
    arm = _arm()
    asyncio.run(
        arm.move_through_joint_positions([_jp(10, 0), _jp(20, 30)], extra=TELEOP)
    )
    assert arm._handle.servo == [
        pytest.approx([math.radians(20), math.radians(30)])
    ]
    assert arm._handle.targets == []
    assert asyncio.run(arm.is_moving()) is False


def test_teleop_move_to_returns_without_waiting():
    arm = _arm()
    asyncio.run(arm.move_to_joint_positions(_jp(45, 0), extra={"waitAtEnd": False}))
    assert arm._handle.targets == [pytest.approx([math.radians(45), 0.0])]


def test_default_still_waits_for_convergence():
    arm = _arm()
    with pytest.raises(TimeoutError):
        asyncio.run(arm.move_through_joint_positions([_jp(20, 30)]))
    with pytest.raises(TimeoutError):
        asyncio.run(arm.move_to_joint_positions(_jp(20, 30)))


def test_raw_setpoint_with_wait_blocks_until_converged():
    arm = _arm()
    with pytest.raises(TimeoutError):
        asyncio.run(
            arm.move_through_joint_positions(
                [_jp(20, 30)], extra={"interpolate": False}
            )
        )
    assert len(arm._handle.targets) == 1


def test_interpolated_no_wait_streams_in_background_until_replaced():
    arm = _arm()

    async def scenario():
        await arm.move_through_joint_positions(
            [_jp(20, 0)], extra={"waitAtEnd": False}
        )
        assert await arm.is_moving()
        await asyncio.sleep(0.05)
        streamed = len(arm._handle.targets)
        assert 0 < streamed < 10  # densified to 2 deg steps at 50 Hz

        await arm.move_through_joint_positions([_jp(5, 5)], extra=TELEOP)
        await asyncio.sleep(0.05)
        # background stream cancelled (a push already in a worker thread may land)
        assert len(arm._handle.targets) <= streamed + 1
        assert arm._handle.servo == [
            pytest.approx([math.radians(5), math.radians(5)])
        ]
        assert not await arm.is_moving()

    asyncio.run(scenario())


def test_stop_cancels_background_stream():
    arm = _arm()

    async def scenario():
        await arm.move_through_joint_positions(
            [_jp(20, 0)], extra={"waitAtEnd": False}
        )
        await arm.stop()
        count = len(arm._handle.targets)
        await asyncio.sleep(0.06)
        assert len(arm._handle.targets) <= count + 1
        assert not await arm.is_moving()
        assert arm._handle.stops == 1

    asyncio.run(scenario())
