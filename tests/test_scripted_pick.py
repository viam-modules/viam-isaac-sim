import asyncio
import itertools
import math
import sys
from types import SimpleNamespace

from google.protobuf.json_format import Parse
from grpclib.const import Status
from grpclib.exceptions import GRPCError

import pytest

from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import MoveRequest

from examples import scripted_pick
from examples.scripted_pick import (
    COMPOSITE_DESCENT_MIN_DURATION_SEC,
    CENTER_NOISE_STDEV_RATIO,
    MIN_SUBTASK_DURATION_SEC,
    EpisodeConfig,
    PickConfig,
    _destination,
    _nearest_symmetric_yaw_degrees,
    _wait_for_firm_grasp,
    _wait_for_gripper_settled,
    _wait_sim_duration,
    execute_pick,
    resolve_pick_region,
    sample_prop_pose,
    sample_center_position,
)
from isaac_module.spatial import ov_to_quat, quat_rotate


class FakeWorld:
    def __init__(self, lifted_z=0.2125, events=None):
        self.lifted_z = lifted_z
        self.events = events
        self.commands = []
        self.sim_time = 0.0

    async def do_command(self, command):
        if command["command"] == "status":
            self.sim_time += 0.25
            return {"sim_time": self.sim_time, "playing": True}
        self.commands.append(command)
        if self.events is not None:
            self.events.append(f"command:{command['command']}")
        if command["command"] == "set_prop_pose":
            return {
                "position": list(command["position"]),
                "orientation_wxyz": list(command["orientation_wxyz"]),
            }
        if command["command"] == "start_episode":
            return {"sample_rate_hz": 15.0}
        if command["command"] == "stop_episode":
            return {"path": "/episodes/000042", "outcome": command["outcome"]}
        return {
            "pose": {
                "position": [0.6, 0.1, self.lifted_z],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        }


class FakeGripper:
    def __init__(
        self,
        grabbed=True,
        holding=True,
        events=None,
        firm_grasps=None,
        moving=None,
        is_moving_error=None,
    ):
        self.grabbed = grabbed
        self.holding = holding
        self.calls = []
        self.events = events
        self._firm_grasps = iter(firm_grasps or ())
        self._last_firm_grasp = holding
        # IsMoving readings in order; the last one repeats. Default: always
        # moving, which keeps the waits on their timeout path.
        self._moving = iter(moving or ())
        self._last_moving = True
        self._is_moving_error = is_moving_error
        self.is_moving_calls = 0

    async def is_moving(self):
        self.is_moving_calls += 1
        if self._is_moving_error is not None:
            raise self._is_moving_error
        try:
            self._last_moving = next(self._moving)
        except StopIteration:
            pass
        return self._last_moving

    async def open(self):
        self.calls.append("open")
        if self.events is not None:
            self.events.append("open")

    async def grab(self):
        self.calls.append("grab")
        if self.events is not None:
            self.events.append("grab")
        return self.grabbed

    async def is_holding_something(self):
        self.calls.append("holding")
        if self.events is not None:
            self.events.append("holding")
        try:
            self._last_firm_grasp = next(self._firm_grasps)
        except StopIteration:
            pass
        return SimpleNamespace(
            is_holding_something=self.holding,
            meta={"firm_grasp": self._last_firm_grasp},
        )


class FakeMotion:
    def __init__(
        self,
        events=None,
        *,
        valid_plan=True,
        invalid_plan_number=None,
    ):
        self.name = "builtin"
        self.destinations = []
        self.plans = []
        self.events = events
        self.valid_plan = valid_plan
        self.invalid_plan_number = invalid_plan_number

    async def get_pose(self, component_name, destination_frame):
        assert component_name == "pick-gripper"
        assert destination_frame == "world"
        return PoseInFrame(
            reference_frame="world",
            pose=Pose(x=250.0, y=0.0, z=300.0, o_z=-1.0),
        )

    async def do_command(self, command):
        if "plan" in command:
            request = Parse(command["plan"], MoveRequest())
            self.plans.append(request)
            if self.events is not None:
                self.events.append(f"plan:{request.destination.pose.z:g}")
            if not self.valid_plan or len(self.plans) == self.invalid_plan_number:
                return {}
            return {"plan": [{"pick-arm": [0.0]}, {"pick-arm": [1.0]}]}
        raise AssertionError(f"unexpected motion command: {command}")

    async def move(self, component_name, destination, **kwargs):
        self.destinations.append((component_name, destination, kwargs))
        if self.events is not None:
            self.events.append(f"move:{destination.pose.z:g}")
        return True

class FakeArm:
    def __init__(
        self, events=None, *, execute_success=True, include_acceleration=True
    ):
        self.events = events
        self.execute_success = execute_success
        self.include_acceleration = include_acceleration
        self.streams = []

    async def do_command(self, command):
        assert command == {"command": "dynamics_metadata"}
        metadata = {"max_velocity_rad_s": [10.0]}
        if self.include_acceleration:
            metadata["max_acceleration_rad_s2"] = [100.0]
        return metadata

    async def move_through_joint_positions_streamed(self, batches):
        async for batch in batches:
            self.streams.append(batch)
            if self.events is not None:
                self.events.append("execute")
            if not self.execute_success:
                raise RuntimeError("could not execute trajectory")
            yield SimpleNamespace()


class FakeMachine:
    async def transform_pose(self, query, destination):
        assert query.reference_frame == "pick-arm:base"
        assert destination == "world"
        return SimpleNamespace(
            pose=SimpleNamespace(
                x=query.pose.x - 235.0,
                y=query.pose.y,
                z=query.pose.z + 5.0,
            )
        )


def _scripted_args(episode_output_dir, *, cosmos_output):
    return SimpleNamespace(
        address="localhost:8080",
        disable_webrtc=False,
        task="Pick up the block",
        world="sim-world",
        gripper="pick-gripper",
        motion="builtin",
        arm="pick-arm",
        prop="pick_block",
        grasp_height_m=0.015,
        yaw_offset_deg=0.0,
        gripper_settle_sec=0.0,
        no_noise=False,
        fixed_camera="scene-cam",
        fixed_camera_2="",
        wrist_camera="wrist-cam",
        episode_output_dir=str(episode_output_dir) if episode_output_dir else None,
        episode_id="000042",
        seed=42,
        cosmos_output=cosmos_output,
    )


def test_pick_region_follows_runtime_arm_base_xy():
    region_min, region_max = asyncio.run(resolve_pick_region(FakeMachine(), "pick-arm"))

    assert region_min == pytest.approx((0.015, -0.12, 0.0))
    assert region_max == pytest.approx((0.145, 0.12, 0.0))


def test_parallel_jaw_yaw_uses_equivalent_short_rotation():
    assert _nearest_symmetric_yaw_degrees(172.0) == pytest.approx(-8.0)
    assert _nearest_symmetric_yaw_degrees(-172.0) == pytest.approx(8.0)
    assert _nearest_symmetric_yaw_degrees(45.0) == pytest.approx(45.0)


def test_square_prop_yaw_uses_the_quarter_turn_symmetry():
    assert _nearest_symmetric_yaw_degrees(40.0, 90.0) == pytest.approx(40.0)
    assert _nearest_symmetric_yaw_degrees(130.0, 90.0) == pytest.approx(40.0)
    assert _nearest_symmetric_yaw_degrees(-139.0, 90.0) == pytest.approx(-49.0 + 90.0)
    for yaw in range(-180, 180, 7):
        assert -45.0 <= _nearest_symmetric_yaw_degrees(float(yaw), 90.0) < 45.0


def test_square_block_grasp_yaw_stays_within_a_quarter_turn():
    for seed in range(20):
        motion = FakeMotion()
        asyncio.run(
            execute_pick(
                FakeWorld(),
                FakeGripper(),
                motion,
                FakeArm(),
                seed,
                config=PickConfig(gripper_settle_sec=0.0, noise=False),
            )
        )
        grasp_thetas = [abs(plan.destination.pose.theta) for plan in motion.plans]
        assert max(grasp_thetas) <= 45.0


def test_downward_destination_preserves_world_yaw():
    destination = _destination(0.6, 0.1, 0.2, 37.0)
    pose = destination.pose
    orientation = ov_to_quat(pose.o_x, pose.o_y, pose.o_z, math.radians(pose.theta))
    tool_x = quat_rotate(orientation, (-1.0, 0.0, 0.0))

    assert math.degrees(math.atan2(tool_x[1], tool_x[0])) == pytest.approx(37.0)


def test_gripper_settle_wait_uses_elapsed_simulator_time(monkeypatch):
    class SimClockWorld:
        def __init__(self):
            self.times = iter((10.0, 10.4, 11.9, 12.0))
            self.status_calls = 0

        async def do_command(self, command):
            assert command == {"command": "status"}
            self.status_calls += 1
            return {"sim_time": next(self.times), "playing": True}

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    world = SimClockWorld()

    asyncio.run(_wait_sim_duration(world, 2.0))

    assert world.status_calls == 4
    assert sleep_calls == pytest.approx([0.05, 0.05, 0.05])



def test_firm_grasp_wait_stops_on_latch_then_settles(monkeypatch):
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    gripper = FakeGripper(firm_grasps=(False, True))

    firm, meta = asyncio.run(_wait_for_firm_grasp(FakeWorld(), gripper, 2.0))

    assert firm is True
    assert meta["firm_grasp"] is True
    assert gripper.calls == ["holding", "holding"]
    assert sleep_calls == pytest.approx([0.05, 0.05])


def test_firm_grasp_wait_times_out_on_simulator_clock(monkeypatch):
    async def fake_sleep(_duration):
        pass

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    gripper = FakeGripper(holding=False, firm_grasps=(False, False))

    firm, meta = asyncio.run(_wait_for_firm_grasp(FakeWorld(), gripper, 0.5))

    assert firm is False
    assert meta["firm_grasp"] is False
    assert gripper.calls == ["holding", "holding"]


def test_firm_grasp_wait_ends_early_when_jaw_closes_without_latching(monkeypatch):
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    # FakeWorld advances 0.25 s per status read, so a 60 s timeout would idle
    # for hundreds of polls; the jaw moving then stopping ends it after two.
    gripper = FakeGripper(holding=False, firm_grasps=(False,), moving=(True, False))

    firm, meta = asyncio.run(_wait_for_firm_grasp(FakeWorld(), gripper, 60.0))

    assert firm is False
    assert meta["firm_grasp"] is False
    assert gripper.calls == ["holding", "holding", "holding"]
    assert gripper.is_moving_calls == 2
    assert sleep_calls == pytest.approx([0.05])


def test_firm_grasp_wait_rechecks_latch_when_jaw_stops(monkeypatch):
    async def fake_sleep(_duration):
        pass

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    # latch flips between the holding read and the IsMoving read
    gripper = FakeGripper(firm_grasps=(False, False, True), moving=(True, False))

    firm, meta = asyncio.run(_wait_for_firm_grasp(FakeWorld(), gripper, 60.0))

    assert firm is True
    assert meta["firm_grasp"] is True
    assert gripper.calls == ["holding", "holding", "holding"]


def test_firm_grasp_wait_keeps_full_timeout_without_is_moving(monkeypatch):
    async def fake_sleep(_duration):
        pass

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    gripper = FakeGripper(
        holding=False,
        firm_grasps=(False,),
        is_moving_error=GRPCError(Status.UNIMPLEMENTED),
    )

    firm, _meta = asyncio.run(_wait_for_firm_grasp(FakeWorld(), gripper, 1.0))

    assert firm is False
    # 0.25 s of simulator time per status read: four polls reach the timeout
    assert gripper.calls == ["holding"] * 4


def test_gripper_settle_wait_ends_when_jaw_moves_then_stops(monkeypatch):
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    gripper = FakeGripper(moving=(True, True, False))

    result = asyncio.run(
        _wait_for_gripper_settled(FakeWorld(), gripper, 3.0, "gripper open wait")
    )

    assert result["settled"] is True
    assert result["fallback"] is False
    assert result["clock"] == "sim"
    assert result["timeout_sec"] == 3.0
    assert 0.0 < result["elapsed_sec"] < 3.0
    assert gripper.is_moving_calls == 3
    assert sleep_calls == pytest.approx([0.05, 0.05])


def test_gripper_settle_wait_waits_full_timeout_when_jaw_never_moves(monkeypatch):
    async def fake_sleep(_duration):
        pass

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    gripper = FakeGripper(moving=(False,))

    result = asyncio.run(
        _wait_for_gripper_settled(FakeWorld(), gripper, 1.0, "gripper open wait")
    )

    assert result["settled"] is False
    assert result["fallback"] is False
    assert result["elapsed_sec"] >= 1.0


def test_gripper_settle_wait_falls_back_to_fixed_wait_without_is_moving(monkeypatch):
    settles = []

    async def record_settle(world, duration_sec):
        settles.append(duration_sec)

    async def fake_sleep(_duration):
        pass

    monkeypatch.setattr(scripted_pick, "_wait_sim_duration", record_settle)
    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    gripper = FakeGripper(is_moving_error=NotImplementedError("no IsMoving"))

    result = asyncio.run(
        _wait_for_gripper_settled(FakeWorld(), gripper, 2.0, "gripper open wait")
    )

    assert result["fallback"] is True
    assert result["settled"] is False
    assert len(settles) == 1
    assert 0.0 < settles[0] <= 2.0


def test_gripper_settle_wait_uses_wall_clock_without_simulator_time(monkeypatch):
    class RealRobotWorld:
        async def do_command(self, command):
            assert command == {"command": "status"}
            return {"booted": True}

    async def fake_sleep(_duration):
        pass

    monkeypatch.setattr(scripted_pick.asyncio, "sleep", fake_sleep)
    gripper = FakeGripper(moving=(True, False))

    result = asyncio.run(
        _wait_for_gripper_settled(RealRobotWorld(), gripper, 5.0, "gripper open wait")
    )

    assert result["settled"] is True
    assert result["clock"] == "wall"
    assert result["elapsed_sec"] < 1.0


def test_zero_settle_timeout_does_not_wait():
    gripper = FakeGripper()

    result = asyncio.run(
        _wait_for_gripper_settled(FakeWorld(), gripper, 0.0, "gripper open wait")
    )

    assert result["settled"] is False
    assert gripper.is_moving_calls == 0


def test_scripted_pick_verifies_lift_and_finishes_while_holding():
    world = FakeWorld()
    gripper = FakeGripper()
    motion = FakeMotion()
    arm = FakeArm()

    result = asyncio.run(
        execute_pick(
            world,
            gripper,
            motion,
            arm,
            42,
            PickConfig(
                pick_region_min_m=(0.01, -0.1, 0.0),
                pick_region_max_m=(0.14, 0.1, 0.0),
                gripper_settle_sec=0.0,
                noise=False,
            ),
        )
    )

    assert result["outcome"] == "success"
    assert gripper.calls == ["open", "grab", "holding", "holding"]
    assert result["holding_meta"] == {"firm_grasp": True}
    timing = result["timing"]
    assert timing["wall_total_sec"] >= 0.0
    assert timing["sim_elapsed_sec"] > 0.0
    assert set(timing["phases_sec"]) == {
        "prop_reset",
        "plan",
        "gripper_open",
        "execute",
        "grab",
        "grasp_settle",
        "lift_check",
    }
    assert [move["label"] for move in timing["moves"]] == [
        "center",
        "descent-0",
        "descent-1",
        "lift",
    ]
    assert all(
        move["planned_sec"] > 0.0 and move["execute_sec"] >= 0.0
        for move in timing["moves"]
    )
    assert timing["gripper_waits"]["open"]["timeout_sec"] == pytest.approx(
        scripted_pick.OPEN_SETTLE_MAX_SEC
    )
    assert len(motion.plans) == 4
    center_request, pre_grasp_request, descent_request, lift_request = motion.plans
    assert center_request.name == "builtin"
    assert center_request.component_name == "pick-gripper"
    block_center_z_mm = scripted_pick.PICK_BLOCK_HEIGHT_M * 500.0
    assert center_request.destination.pose.z == pytest.approx(
        block_center_z_mm + PickConfig.approach_clearance_m * 1000.0
    )
    assert center_request.destination.pose.theta == pytest.approx(0.0)
    center_constraint = center_request.constraints.orientation_constraint[0]
    assert 0.0 < center_constraint.orientation_tolerance_degs <= 30.0
    assert [
        request.destination.pose.z
        for request in (pre_grasp_request, descent_request, lift_request)
    ] == pytest.approx(
        [
            block_center_z_mm + scripted_pick.PRE_GRASP_CLEARANCE_M * 1000.0,
            block_center_z_mm,
            block_center_z_mm + PickConfig.lift_height_m * 1000.0,
        ]
    )
    assert all(
        request.constraints.linear_constraint[0].line_tolerance_mm == 5.0
        for request in (pre_grasp_request, descent_request, lift_request)
    )
    assert motion.destinations == []
    assert len(arm.streams) == 4
    center_points, pre_grasp_points, descent_points, lift_points = arm.streams
    assert center_points[-1].time.total_seconds() >= MIN_SUBTASK_DURATION_SEC
    assert pre_grasp_points[-1].time.total_seconds() >= MIN_SUBTASK_DURATION_SEC
    assert (
        descent_points[-1].time.total_seconds() + 1e-6
        >= COMPOSITE_DESCENT_MIN_DURATION_SEC
    )
    assert (
        descent_points[-1].time.total_seconds()
        + lift_points[-1].time.total_seconds()
        >= MIN_SUBTASK_DURATION_SEC
    )
    for points in arm.streams:
        assert points[0].time.total_seconds() == 0.0
        assert points[0].constraints.velocities == pytest.approx([0.0])
        assert points[-1].constraints.velocities == pytest.approx([0.0])
        assert (
            max(
                (right.time - left.time).total_seconds()
                for left, right in itertools.pairwise(points)
            )
            <= 1.0 / 20.0 + 1e-9
        )
    assert world.commands[-1] == {"command": "get_prop_pose", "prop": "pick_block"}
    # Verify set_prop_pose was called with correct structure
    assert world.commands[0]["command"] == "set_prop_pose"
    assert world.commands[0]["prop"] == "pick_block"
    assert "position" in world.commands[0]
    assert "orientation_wxyz" in world.commands[0]
    assert world.commands[0]["position"][2] == pytest.approx(
        scripted_pick.PICK_BLOCK_HEIGHT_M / 2.0
    )


def test_scripted_pick_adds_seeded_noise_and_correction_waypoints():
    motion = FakeMotion()
    arm = FakeArm()

    result = asyncio.run(
        execute_pick(
            FakeWorld(),
            FakeGripper(),
            motion,
            arm,
            42,
            PickConfig(gripper_settle_sec=0.0),
        )
    )

    assert scripted_pick.GRASP_HEIGHT_RANGE_M[0] <= result["grasp_height_m"]
    assert result["grasp_height_m"] <= scripted_pick.GRASP_HEIGHT_RANGE_M[1]
    assert result["grasp_height_m"] == scripted_pick.grasp_height(42)
    assert len(motion.destinations) == 1
    start_pose = motion.destinations[0][1].pose
    assert (start_pose.x, start_pose.y, start_pose.z) != (250.0, 0.0, 300.0)

    assert len(motion.plans) == 8
    (
        _center_midpoint,
        _center_arrival,
        approach_midpoint,
        correction_arrival,
        rotation_corrected,
        position_corrected,
        grasp,
        lift,
    ) = (request.destination.pose for request in motion.plans)
    grasp_z_mm = result["grasp_height_m"] * 1000.0
    correction_z_mm = grasp_z_mm + scripted_pick.PRE_GRASP_CLEARANCE_M * 1000.0
    assert approach_midpoint.z > correction_z_mm
    assert (
        correction_arrival.x,
        correction_arrival.y,
        correction_arrival.z,
    ) == pytest.approx(
        (rotation_corrected.x, rotation_corrected.y, rotation_corrected.z)
    )
    assert correction_arrival.z == pytest.approx(correction_z_mm)
    assert correction_arrival.theta != pytest.approx(rotation_corrected.theta)
    assert rotation_corrected.theta == pytest.approx(position_corrected.theta)
    target_x_m, target_y_m, initial_z_m = result["target_pose"]["position"]
    assert (position_corrected.x, position_corrected.y, position_corrected.z) == (
        pytest.approx(target_x_m * 1000.0),
        pytest.approx(target_y_m * 1000.0),
        pytest.approx(correction_z_mm),
    )
    assert (grasp.x, grasp.y, grasp.z) == (
        pytest.approx(target_x_m * 1000.0),
        pytest.approx(target_y_m * 1000.0),
        pytest.approx(grasp_z_mm),
    )
    assert (lift.x, lift.y, lift.z) != pytest.approx(
        (target_x_m * 1000.0, target_y_m * 1000.0, (initial_z_m + 0.2) * 1000.0)
    )
    approach_midpoint_points, correction_arrival_points = arm.streams[2:4]
    rotation_correction_points, position_correction_points = arm.streams[4:6]
    final_descent_points = arm.streams[6]
    assert (
        approach_midpoint_points[-1].time.total_seconds() + 1e-6
        >= scripted_pick.APPROACH_WAYPOINT_MIN_DURATION_SEC
    )
    assert (
        correction_arrival_points[-1].time.total_seconds() + 1e-6
        >= scripted_pick.APPROACH_WAYPOINT_MIN_DURATION_SEC
    )
    correction_durations = [
        rotation_correction_points[-1].time.total_seconds(),
        position_correction_points[-1].time.total_seconds(),
    ]
    assert all(
        duration >= scripted_pick.CORRECTION_MOVE_MIN_DURATION_SEC
        for duration in correction_durations
    )
    final_descent_duration = final_descent_points[-1].time.total_seconds()
    assert final_descent_duration + 1e-6 >= scripted_pick.FINAL_DESCENT_MIN_DURATION_SEC
    assert final_descent_duration > max(correction_durations)
    acceleration_limit_deg_s2 = math.degrees(100.0)
    assert all(
        abs(acceleration) <= acceleration_limit_deg_s2 + 1e-6
        for points in arm.streams
        for point in points
        for acceleration in point.constraints.accelerations
    )


def test_scripted_pick_uses_acceleration_fallback_for_older_arm_metadata():
    arm = FakeArm(include_acceleration=False)

    asyncio.run(
        execute_pick(
            FakeWorld(),
            FakeGripper(),
            FakeMotion(),
            arm,
            42,
            PickConfig(gripper_settle_sec=0.0, noise=False),
        )
    )

    assert all(
        abs(acceleration)
        <= scripted_pick.DEFAULT_MAX_JOINT_ACCELERATION_DEG_S2 * 1.02
        for points in arm.streams
        for point in points
        for acceleration in point.constraints.accelerations
    )


def test_scripted_pick_reports_failed_grab_without_lift_query():
    world = FakeWorld()
    events = []
    gripper = FakeGripper(grabbed=False, events=events)
    motion = FakeMotion(events=events)
    arm = FakeArm(events=events)

    result = asyncio.run(
        execute_pick(
            world,
            gripper,
            motion,
            arm,
            7,
            PickConfig(
                minimum_lift_m=0.05,
                gripper_settle_sec=0.0,
                noise=False,
            ),
        )
    )

    assert result["outcome"] == "failure"
    assert result["grabbed"] is False
    assert gripper.calls == ["open", "grab"]
    assert [move["label"] for move in result["timing"]["moves"]] == [
        "center",
        "descent-0",
        "descent-1",
        "retreat",
    ]
    assert "grasp_settle" not in result["timing"]["phases_sec"]
    assert len(motion.plans) == 4
    block_center_z_mm = scripted_pick.PICK_BLOCK_HEIGHT_M * 500.0
    assert [request.destination.pose.z for request in motion.plans] == pytest.approx(
        [
            block_center_z_mm + PickConfig.approach_clearance_m * 1000.0,
            block_center_z_mm + scripted_pick.PRE_GRASP_CLEARANCE_M * 1000.0,
            block_center_z_mm,
            block_center_z_mm + PickConfig.approach_clearance_m * 1000.0,
        ]
    )
    assert motion.destinations == []
    assert len(arm.streams) == 4
    assert [command["command"] for command in world.commands] == ["set_prop_pose"]
    assert events == [
        f"plan:{block_center_z_mm + PickConfig.approach_clearance_m * 1000.0:g}",
        "open",
        "execute",
        f"plan:{block_center_z_mm + scripted_pick.PRE_GRASP_CLEARANCE_M * 1000.0:g}",
        "execute",
        f"plan:{block_center_z_mm:g}",
        "execute",
        "grab",
        f"plan:{block_center_z_mm + PickConfig.approach_clearance_m * 1000.0:g}",
        "execute",
    ]


def test_center_plan_failure_stops_before_capture():
    world = FakeWorld()

    with pytest.raises(RuntimeError, match="returned an invalid plan"):
        asyncio.run(
            execute_pick(
                world,
                FakeGripper(),
                FakeMotion(valid_plan=False),
                FakeArm(),
                42,
                config=PickConfig(gripper_settle_sec=0.0),
                episode=EpisodeConfig("/episodes", "000042"),
            )
        )

    assert [command["command"] for command in world.commands] == ["set_prop_pose"]


def test_center_execute_failure_finalizes_incomplete_capture():
    world = FakeWorld()

    with pytest.raises(RuntimeError, match="could not execute trajectory"):
        asyncio.run(
            execute_pick(
                world,
                FakeGripper(),
                FakeMotion(),
                FakeArm(execute_success=False),
                42,
                config=PickConfig(gripper_settle_sec=0.0),
                episode=EpisodeConfig("/episodes", "000042"),
            )
        )

    assert world.commands[-1] == {"command": "stop_episode", "outcome": "incomplete"}


def test_pre_grasp_plan_failure_resumes_before_finalizing_capture():
    events = []
    world = FakeWorld(events=events)

    with pytest.raises(RuntimeError, match="returned an invalid plan"):
        asyncio.run(
            execute_pick(
                world,
                FakeGripper(events=events),
                FakeMotion(events=events, invalid_plan_number=2),
                FakeArm(events=events),
                42,
                config=PickConfig(gripper_settle_sec=0.0, noise=False),
                episode=EpisodeConfig("/episodes", "000042"),
            )
        )

    pause_index = events.index("command:pause")
    assert events[pause_index : pause_index + 4] == [
        "command:pause",
        f"plan:{scripted_pick.PICK_BLOCK_HEIGHT_M * 500.0 + scripted_pick.PRE_GRASP_CLEARANCE_M * 1000.0:g}",
        "command:play",
        "command:stop_episode",
    ]
    assert world.commands[-1] == {"command": "stop_episode", "outcome": "incomplete"}


def test_center_setup_finishes_before_capture(monkeypatch):
    events = []

    async def record_settle(world, gripper, timeout_sec, label):
        events.append(f"settle:{timeout_sec:g}")
        return {"settled": True}

    monkeypatch.setattr(scripted_pick, "_wait_for_gripper_settled", record_settle)
    asyncio.run(
        execute_pick(
            FakeWorld(events=events),
            FakeGripper(events=events),
            FakeMotion(events=events),
            FakeArm(events=events),
            42,
            config=PickConfig(gripper_settle_sec=2.0),
            episode=EpisodeConfig("/episodes", "000042"),
        )
    )

    center_plan_index = next(
        index for index, event in enumerate(events) if event.startswith("plan:")
    )
    assert center_plan_index < events.index("open")
    assert events.index("open") < events.index("settle:0.5")
    assert events.index("settle:0.5") < events.index("command:start_episode")
    assert events.index("command:start_episode") < events.index(
        "command:mark_episode_stage"
    )
    assert events.index("command:mark_episode_stage") < events.index("execute")


def test_scripted_pick_can_wrap_action_in_native_episode_capture():
    events = []
    world = FakeWorld(events=events)
    result = asyncio.run(
        execute_pick(
            world,
            FakeGripper(events=events),
            FakeMotion(events=events),
            FakeArm(events=events),
            42,
            config=PickConfig(gripper_settle_sec=0.0, noise=False),
            episode=EpisodeConfig(
                "/episodes",
                "000042",
                upload={"dataset_id": "dataset-1", "tags": ["run"]},
            ),
        )
    )

    center_plan_index = next(
        index for index, event in enumerate(events) if event.startswith("plan:")
    )
    assert center_plan_index < events.index("open")
    assert events.index("open") < events.index("command:start_episode")
    assert events.index("command:start_episode") < events.index("execute")
    assert result["capture"] == {
        "path": "/episodes/000042",
        "outcome": "success",
    }
    assert [command["command"] for command in world.commands] == [
        "set_prop_pose",
        "start_episode",
        "mark_episode_stage",
        "mark_episode_stage",
        "pause",
        "play",
        "pause",
        "mark_episode_stage",
        "play",
        "pause",
        "play",
        "get_prop_pose",
        "mark_episode_stage",
        "stop_episode",
    ]
    pause_index = events.index("command:pause")
    assert events[pause_index - 1 : pause_index + 4] == [
        "command:mark_episode_stage",
        "command:pause",
        f"plan:{scripted_pick.PICK_BLOCK_HEIGHT_M * 500.0 + scripted_pick.PRE_GRASP_CLEARANCE_M * 1000.0:g}",
        "command:play",
        "execute",
    ]
    assert world.commands[1]["task"] == scripted_pick.DEFAULT_PROMPT
    assert [
        command["task"]
        for command in world.commands
        if command["command"] == "mark_episode_stage"
    ] == scripted_pick.STAGE_PROMPTS
    assert world.commands[1]["target"] == {
        "prop_name": "pick_block",
        # Position should match what was set via set_prop_pose
        "position_m": world.commands[0]["position"],
        # Rotation should match what was set via set_prop_pose
        "rotation_wxyz": world.commands[0]["orientation_wxyz"],
    }
    assert world.commands[1]["target"]["position_m"][2] == pytest.approx(
        scripted_pick.PICK_BLOCK_HEIGHT_M / 2.0
    )
    # Verify rotation is valid unit quaternion
    w, x, y, z = world.commands[1]["target"]["rotation_wxyz"]
    assert math.isclose(w * w + x * x + y * y + z * z, 1.0, abs_tol=1e-9)
    assert world.commands[1]["output_dir"] == "/episodes"
    assert world.commands[1]["cosmos_output_dir"] is None
    assert world.commands[1]["fixed_camera_2"] == ""
    assert world.commands[-1] == {
        "command": "stop_episode",
        "outcome": "success",
        "upload": {"dataset_id": "dataset-1", "tags": ["run"]},
    }


def test_grip_only_captures_descent_grip_and_short_lift():
    events = []
    world = FakeWorld(lifted_z=0.0625, events=events)
    gripper = FakeGripper(events=events)
    motion = FakeMotion(events=events)

    result = asyncio.run(
        execute_pick(
            world,
            gripper,
            motion,
            FakeArm(events=events),
            42,
            config=PickConfig(
                grip_only=True,
                grip_lift_m=0.05,
                gripper_settle_sec=0.0,
                noise=False,
            ),
            episode=EpisodeConfig("/episodes", "000042"),
        )
    )

    assert result["outcome"] == "success"
    assert gripper.calls == ["open", "grab", "holding", "holding"]
    assert set(result["timing"]["phases_sec"]) == {
        "prop_reset",
        "gripper_open",
        "pre_grasp_move",
        "capture_start",
        "grasp_move",
        "grab",
        "grasp_settle",
        "lift_move",
        "lift_check",
        "capture_stop",
    }
    assert result["timing"]["moves"] == []
    grasp_z_mm = scripted_pick.PICK_BLOCK_HEIGHT_M * 500.0
    grip_lift_z_mm = grasp_z_mm + 50.0
    assert [entry[1].pose.z for entry in motion.destinations] == pytest.approx(
        [grip_lift_z_mm, grasp_z_mm, grip_lift_z_mm]
    )
    assert motion.destinations[0][2]["constraints"] is None
    assert all(
        entry[2]["constraints"].linear_constraint[0].line_tolerance_mm == 5.0
        for entry in motion.destinations[1:]
    )
    assert events == [
        "command:set_prop_pose",
        "open",
        f"move:{grip_lift_z_mm:g}",
        "command:start_episode",
        "command:mark_episode_stage",
        f"move:{grasp_z_mm:g}",
        "command:mark_episode_stage",
        "grab",
        "holding",
        f"move:{grip_lift_z_mm:g}",
        "holding",
        "command:get_prop_pose",
        "command:stop_episode",
    ]


def test_grip_only_ascends_after_failed_grab():
    gripper = FakeGripper(grabbed=False)
    motion = FakeMotion()

    result = asyncio.run(
        execute_pick(
            FakeWorld(),
            gripper,
            motion,
            FakeArm(),
            42,
            config=PickConfig(grip_only=True, gripper_settle_sec=0.0, noise=False),
        )
    )

    assert result["outcome"] == "failure"
    assert gripper.calls == ["open", "grab"]
    grasp_z_mm = scripted_pick.PICK_BLOCK_HEIGHT_M * 500.0
    grip_lift_z_mm = grasp_z_mm + PickConfig.grip_lift_m * 1000.0
    assert [entry[1].pose.z for entry in motion.destinations] == pytest.approx(
        [grip_lift_z_mm, grasp_z_mm, grip_lift_z_mm]
    )


def test_scripted_pick_native_only_uses_run_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_root = "run"

    episode = scripted_pick._episode_config(
        _scripted_args(run_root, cosmos_output=False)
    )

    assert episode is not None
    assert episode.output_dir == str(tmp_path / "run" / "episodes")
    assert episode.cosmos_output_dir is None


def test_scripted_pick_cosmos_uses_run_root_and_sibling_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_root = "run"
    episode = scripted_pick._episode_config(
        _scripted_args(run_root, cosmos_output=True)
    )

    assert episode is not None
    assert episode.output_dir == str(tmp_path / "run" / "episodes")
    assert episode.cosmos_output_dir == str(
        tmp_path / "run" / "cosmoswriter" / "000042"
    )

    world = FakeWorld()
    asyncio.run(
        execute_pick(
            world,
            FakeGripper(),
            FakeMotion(),
            FakeArm(),
            42,
            config=PickConfig(gripper_settle_sec=0.0, noise=False),
            episode=episode,
        )
    )

    assert [command["command"] for command in world.commands] == [
        "set_prop_pose",
        "start_episode",
        "mark_episode_stage",
        "mark_episode_stage",
        "pause",
        "play",
        "pause",
        "mark_episode_stage",
        "play",
        "pause",
        "play",
        "get_prop_pose",
        "mark_episode_stage",
        "stop_episode",
    ]
    assert world.commands[1]["output_dir"] == str(tmp_path / "run" / "episodes")
    assert world.commands[1]["cosmos_output_dir"] == str(
        tmp_path / "run" / "cosmoswriter" / "000042"
    )


def test_scripted_pick_configures_optional_second_fixed_camera(tmp_path):
    args = _scripted_args(tmp_path, cosmos_output=False)
    args.fixed_camera_2 = "third-cam"

    episode = scripted_pick._episode_config(args)

    assert episode is not None
    assert episode.fixed_camera_2_name == "third-cam"


def test_scripted_pick_rejects_cosmos_without_native_run_root():
    with pytest.raises(
        ValueError, match="--cosmos-output requires --episode-output-dir"
    ):
        scripted_pick._episode_config(_scripted_args(None, cosmos_output=True))


def test_scripted_pick_cli_rejects_caller_selected_cosmos_root(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripted_pick.py",
            "localhost:8080",
            "--episode-output-dir",
            "/run",
            "--cosmos-output-dir",
            "/cosmos",
        ],
    )

    with pytest.raises(SystemExit):
        scripted_pick.main()


def test_sample_prop_pose_is_deterministic():
    """Same seed and config produces identical pose."""
    region_min = (0.25, -0.12, 0.0)
    region_max = (0.38, 0.12, 0.0)
    yaw_range = (-180.0, 180.0)
    width, depth, height = 0.075, 0.015, 0.025

    result1 = sample_prop_pose(
        42, region_min, region_max, yaw_range, width, depth, height
    )
    result2 = sample_prop_pose(
        42, region_min, region_max, yaw_range, width, depth, height
    )

    assert result1 == result2


def test_center_position_noise_is_seeded_and_scaled_from_wrist_height():
    target = (0.3, -0.1, 0.2)

    sampled = sample_center_position(42, target)

    assert sampled == sample_center_position(42, target)
    assert sampled != target
    standard_deviation = target[2] * CENTER_NOISE_STDEV_RATIO
    expected_rng = scripted_pick.random.Random("center-arrival:42")
    assert sampled == pytest.approx(
        tuple(expected_rng.gauss(value, standard_deviation) for value in target)
    )


def test_center_position_at_zero_height_has_no_noise():
    assert sample_center_position(42, (0.3, -0.1, 0.0)) == (0.3, -0.1, 0.0)


def test_sample_prop_pose_respects_region_bounds():
    """Sampled position is within region accounting for rotated footprint."""
    region_min = (0.25, -0.12, 0.0)
    region_max = (0.38, 0.12, 0.0)
    yaw_range = (-180.0, 180.0)
    width, depth, height = 0.075, 0.015, 0.025

    position, _ = sample_prop_pose(
        123, region_min, region_max, yaw_range, width, depth, height
    )
    x, y, z = position

    # Account for rotated footprint at extremes
    yaw_rad = math.radians(0)  # worst case footprint
    half_x = (abs(math.cos(yaw_rad)) * width + abs(math.sin(yaw_rad)) * depth) / 2.0
    half_y = (abs(math.sin(yaw_rad)) * width + abs(math.cos(yaw_rad)) * depth) / 2.0

    assert region_min[0] + half_x <= x <= region_max[0] - half_x
    assert region_min[1] + half_y <= y <= region_max[1] - half_y


def test_sample_prop_pose_z_at_support_plus_half_height():
    """Z coordinate is at support plane + half prop height."""
    region_min = (0.1, -0.2, 0.05)  # non-zero support plane
    region_max = (0.4, 0.2, 0.05)
    yaw_range = (0.0, 0.0)
    width, depth, height = 0.075, 0.015, 0.025

    position, _ = sample_prop_pose(
        0, region_min, region_max, yaw_range, width, depth, height
    )

    # Z should be support plane + half height
    assert position[2] == pytest.approx(region_min[2] + height / 2.0)


def test_sample_prop_pose_returns_valid_quaternion():
    """Orientation is a valid unit quaternion."""
    region_min = (0.25, -0.12, 0.0)
    region_max = (0.38, 0.12, 0.0)
    yaw_range = (-180.0, 180.0)
    width, depth, height = 0.075, 0.015, 0.025

    _, orientation = sample_prop_pose(
        999, region_min, region_max, yaw_range, width, depth, height
    )
    w, x, y, z = orientation

    # Verify unit quaternion
    assert math.isclose(w * w + x * x + y * y + z * z, 1.0, abs_tol=1e-9)


def test_distractor_is_placed_clear_of_the_target_and_seeded():
    config = PickConfig(distractor_prop="blue_block")
    target = (0.10, 0.0, 0.015)
    for seed in range(50):
        position, _orientation = scripted_pick.sample_distractor_pose(seed, config, target)
        assert math.dist(position[:2], target[:2]) >= scripted_pick.DISTRACTOR_MIN_SEPARATION_M
        assert scripted_pick.sample_distractor_pose(seed, config, target)[0] == position


def test_colored_pick_places_the_distractor_and_names_the_target_in_stages():
    events = []
    world = FakeWorld(events=events)
    asyncio.run(
        execute_pick(
            world,
            FakeGripper(events=events),
            FakeMotion(events=events),
            FakeArm(events=events),
            42,
            config=PickConfig(
                gripper_settle_sec=0.0,
                noise=False,
                distractor_prop="blue_block",
                target_label="red",
            ),
            episode=EpisodeConfig("/episodes", "000042", task="pick up the red block"),
        )
    )

    placed = [c["prop"] for c in world.commands if c["command"] == "set_prop_pose"]
    assert placed == ["pick_block", "blue_block"]
    stages = [c["task"] for c in world.commands if c["command"] == "mark_episode_stage"]
    assert stages == scripted_pick.stage_prompts("red")
    assert all("the block" not in stage for stage in stages)
    assert any("the red block" in stage for stage in stages)


class FakePlaceWorld(FakeWorld):
    """Reports the block wherever the test says it ended up after release."""

    def __init__(self, placed_position, events=None):
        super().__init__(events=events)
        self.placed_position = placed_position
        self.pad_position = None

    async def do_command(self, command):
        result = await super().do_command(command)
        if command["command"] == "set_prop_pose" and command["prop"] == "place_pad":
            self.pad_position = command["position"]
        gets = [c for c in self.commands if c["command"] == "get_prop_pose"]
        if command["command"] == "get_prop_pose" and len(gets) == 2:
            return {"pose": {"position": self.placed_position(self.pad_position)}}
        return result


def _run_place(placed_position, holding_after_release=False):
    world = FakePlaceWorld(placed_position)
    gripper = FakeGripper()
    opens = []

    async def open_and_release():
        opens.append(True)
        if len(opens) > 1:
            gripper.holding = holding_after_release

    gripper.open = open_and_release
    result = asyncio.run(
        execute_pick(
            world,
            gripper,
            FakeMotion(),
            FakeArm(),
            42,
            config=PickConfig(
                gripper_settle_sec=0.0,
                noise=False,
                place_prop="place_pad",
                place_label="green",
            ),
            episode=EpisodeConfig("/episodes", "000042", task="put the block on the green pad"),
        )
    )
    return world, result


def _resting_on(pad):
    rest_z = pad[2] + scripted_pick.PLACE_PAD_HEIGHT_M / 2 + scripted_pick.PICK_BLOCK_HEIGHT_M / 2
    return [pad[0] + 0.01, pad[1] - 0.01, rest_z]


def test_place_pad_is_axis_aligned_clear_of_the_block_and_seeded():
    config = PickConfig(place_prop="place_pad")
    target = (0.10, 0.0, 0.015)
    for seed in range(50):
        position, orientation = scripted_pick.sample_place_pose(seed, config, target)
        assert math.dist(position[:2], target[:2]) >= scripted_pick.PLACE_MIN_SEPARATION_M
        assert orientation == (1.0, 0.0, 0.0, 0.0)
        assert scripted_pick.sample_place_pose(seed, config, target)[0] == position


def test_block_on_pad_checks_the_outline_and_resting_height():
    pad = (0.3, 0.0, 0.001)
    height = scripted_pick.PICK_BLOCK_HEIGHT_M
    resting = _resting_on(pad)
    assert scripted_pick.block_on_pad(resting, pad, height)
    assert not scripted_pick.block_on_pad([pad[0] + 0.05, pad[1], resting[2]], pad, height)
    assert not scripted_pick.block_on_pad([pad[0], pad[1], resting[2] + 0.05], pad, height)


def test_place_episode_puts_the_block_on_the_pad_with_place_stages():
    world, result = _run_place(_resting_on)

    placed = [c["prop"] for c in world.commands if c["command"] == "set_prop_pose"]
    assert placed == ["pick_block", "place_pad"]
    stages = [c["task"] for c in world.commands if c["command"] == "mark_episode_stage"]
    assert stages == scripted_pick.stage_prompts(place_label="green")
    assert stages[3:] == [
        "carry the block above the green pad",
        "lower the block onto the green pad and open the gripper",
        "lift the gripper away from the block",
        "hold still",
    ]
    assert result["outcome"] == "success"


def test_place_episode_fails_when_the_block_misses_or_stays_held():
    _world, missed = _run_place(lambda pad: [pad[0] + 0.2, pad[1], 0.0157])
    assert missed["outcome"] == "failure"
    _world, held = _run_place(_resting_on, holding_after_release=True)
    assert held["outcome"] == "failure"
