"""Moves honor the `extra` keys the RDK teleop pipeline sends:
{"waitAtEnd": false, "interpolate": false} (viam-ufactory-xarm semantics)."""

import asyncio
import math

import pytest
from viam.components.arm import JointPositions

from isaac_module.models.arm import IsaacArm

TELEOP = {"waitAtEnd": False, "interpolate": False}


class _Handle:
    """Records targets; joints never move, so any convergence wait would time out."""

    def __init__(self):
        self.targets = []
        self.paths = []
        self.servo = []
        self.stops = 0
        self.active = False

    def set_joint_targets(self, positions):
        self.targets.append(list(positions))

    def follow_path(self, waypoints):
        self.paths.append([list(w) for w in waypoints])
        self.active = True

    def path_active(self):
        return self.active

    def servo_target(self, positions):
        self.servo.append(list(positions))

    def get_joint_positions(self):
        return [0.0, 0.0]

    def wait_for_joint_targets(self, targets, tolerance):
        from concurrent.futures import Future

        return Future()  # never resolves

    def stop(self):
        self.stops += 1
        self.active = False


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


def test_default_move_through_hands_the_sim_every_waypoint():
    arm = _arm()
    with pytest.raises(TimeoutError):
        asyncio.run(arm.move_through_joint_positions([_jp(10, 0), _jp(20, 30)]))
    assert arm._handle.paths == [
        [
            pytest.approx([math.radians(10), 0.0]),
            pytest.approx([math.radians(20), math.radians(30)]),
        ]
    ]


def test_raw_setpoint_with_wait_goes_to_the_last_waypoint_and_waits():
    arm = _arm()
    with pytest.raises(TimeoutError):
        asyncio.run(
            arm.move_through_joint_positions(
                [_jp(10, 0), _jp(20, 30)], extra={"interpolate": False}
            )
        )
    assert arm._handle.paths == [
        [pytest.approx([math.radians(20), math.radians(30)])]
    ]


def test_no_wait_path_returns_at_once_and_reports_moving_until_stopped():
    arm = _arm()

    async def scenario():
        await arm.move_through_joint_positions(
            [_jp(10, 0), _jp(20, 0)], extra={"waitAtEnd": False}
        )
        assert len(arm._handle.paths[0]) == 2
        assert await arm.is_moving()
        await arm.stop()
        assert not await arm.is_moving()
        assert arm._handle.stops == 1

    asyncio.run(scenario())
