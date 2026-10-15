"""
Run one deterministic scripted pick through normal Viam APIs.
this is an oracle i.e. it's allowed to use perfect information about its environment for planning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from contextlib import contextmanager
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import toppra as ta
import toppra.algorithm as ta_algorithm
import toppra.constraint as ta_constraint

from google.protobuf.json_format import MessageToJson
from grpclib.exceptions import GRPCError

from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import (
    Constraints,
    LinearConstraint,
    MoveRequest,
    OrientationConstraint,
)
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.services.motion import MotionClient

HOME_JOINT_DEGREES = (0.0, 0.0, 90.0, 0.0, 90.0, 0.0)
PICK_REGION_MIN_ARM_M = (0.25, -0.12, 0.0)
PICK_REGION_MAX_ARM_M = (0.38, 0.12, 0.0)
PICK_YAW_RANGE_DEG = (-180.0, 180.0)
DEFAULT_GRIPPER_SETTLE_SEC = 3.0
PRE_GRASP_CLEARANCE_M = 0.05
GRIP_ONLY_LIFT_TOLERANCE_M = 0.005
CAPTURE_SAMPLE_RATE_HZ = 15.0
ACTION_CHUNK_FRAMES = 33
TRAJECTORY_COMMAND_HZ = 20.0
DEFAULT_MAX_JOINT_ACCELERATION_DEG_S2 = 60.0
# TODO: Read sample_rate_hz and action chunk size from configured capture/policy metadata.
MIN_SUBTASK_DURATION_SEC = ACTION_CHUNK_FRAMES / CAPTURE_SAMPLE_RATE_HZ
COMPOSITE_DESCENT_MIN_DURATION_SEC = MIN_SUBTASK_DURATION_SEC / 3
APPROACH_WAYPOINT_MIN_DURATION_SEC = MIN_SUBTASK_DURATION_SEC / 2
CORRECTION_MOVE_MIN_DURATION_SEC = COMPOSITE_DESCENT_MIN_DURATION_SEC / 4
FINAL_DESCENT_MIN_DURATION_SEC = COMPOSITE_DESCENT_MIN_DURATION_SEC
FIRM_GRASP_POLL_SEC = 0.05
FIRM_GRASP_SETTLE_SEC = 0.25
GRIPPER_SETTLE_POLL_SEC = 0.05
# Opening never contacts anything, so its settle wait is bounded short.
OPEN_SETTLE_MAX_SEC = 0.5
START_NOISE_STDEV_RATIO = 0.10
CENTER_MIDPOINT_NOISE_STDEV_RATIO = START_NOISE_STDEV_RATIO
CENTER_NOISE_STDEV_RATIO = 0.05
DESCENT_MIDPOINT_NOISE_STDEV_RATIO = 0.05
DESCENT_ARRIVAL_POSITION_STDEV_M = 0.01
DESCENT_ARRIVAL_ROTATION_STDEV_DEG = 15.0
LIFT_NOISE_STDEV_RATIO = START_NOISE_STDEV_RATIO
# Wide Lite6 fingertip gaps are 22.4 mm closed and 40.2 mm open. A 31.3 mm
# cube is centered between those limits, leaving half of the jaw travel on
# either side of contact.
WIDE_JAW_CLOSED_GAP_M = 0.0224
WIDE_JAW_OPEN_GAP_M = 0.0402
PICK_BLOCK_WIDTH_M = 0.0313
PICK_BLOCK_DEPTH_M = PICK_BLOCK_WIDTH_M
PICK_BLOCK_HEIGHT_M = PICK_BLOCK_WIDTH_M
GRASP_HEIGHT_RANGE_M = (0.01252, PICK_BLOCK_HEIGHT_M / 2.0)

LINEAR_PICK_CONSTRAINTS = Constraints(
    linear_constraint=[
        LinearConstraint(line_tolerance_mm=5.0, orientation_tolerance_degs=2.0)
    ]
)
CENTER_MOVE_CONSTRAINTS = Constraints(
    orientation_constraint=[OrientationConstraint(orientation_tolerance_degs=5.0)]
)


DEFAULT_PROMPT = "pick up the block"


STAGE_PROMPTS = [
    "center the gripper above the block",
    "move the gripper into position close above the block. in the wrist camera the block should be near the bottom center",
    "descend, grip, wait for a firm grip, and lift the block",
    # inference doesn't have a completion signal, but we can detect this hover in code and use that as a stop condition
    "hold still",
]
# A second block placed for target selection stays at least this far (center to
# center) from the target, so the open gripper doesn't touch it.
DISTRACTOR_MIN_SEPARATION_M = 0.08
DISTRACTOR_PLACEMENT_TRIES = 64
# Place stages replace the final "hold still" of a pick; {pad} is the pad label.
PLACE_STAGE_PROMPTS = [
    "carry the block above the {pad} pad",
    "lower the block onto the {pad} pad and open the gripper",
    "lift the gripper away from the block",
    "hold still",
]
# The pad is a flat square kept axis-aligned so "inside the pad" is a box check.
PLACE_PAD_WIDTH_M = 0.08
PLACE_PAD_HEIGHT_M = 0.002
# Pad center to block center: the 4 cm pad half-width plus the block's ~2.2 cm
# half-diagonal plus a gap, so the block starts off the pad. 10 cm left no room
# in the 13 cm-deep pick region for ~17% of seeds.
PLACE_MIN_SEPARATION_M = 0.07
# The block is let go this far above the pad instead of pressed into it.
PLACE_RELEASE_CLEARANCE_M = 0.005
PLACE_RELEASE_SETTLE_SEC = 0.25
# A placed block must rest within this height of sitting flat on the pad.
PLACE_REST_TOLERANCE_M = 0.01


def stage_prompts(target_label: str = "", place_label: str = "") -> list[str]:
    """Stage texts, naming the target block (e.g. "the red block") when labeled,
    and ending with the place stages when a pad label is given."""
    prompts = list(STAGE_PROMPTS)
    if place_label:
        prompts = prompts[:-1] + [
            prompt.format(pad=place_label) for prompt in PLACE_STAGE_PROMPTS
        ]
    if not target_label:
        return prompts
    return [
        prompt.replace("the block", f"the {target_label} block") for prompt in prompts
    ]


def _yaw_to_quaternion_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    """Convert yaw angle (degrees) to WXYZ quaternion (rotation around Z axis)."""
    half = math.radians(yaw_deg) / 2.0
    return math.cos(half), 0.0, 0.0, math.sin(half)


def sample_prop_pose(
    seed: int,
    region_min: tuple[float, float, float],
    region_max: tuple[float, float, float],
    yaw_range_deg: tuple[float, float],
    prop_width: float,
    prop_depth: float,
    prop_height: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Sample a deterministic prop pose within the workspace.

    Uses a seeded random generator for reproducibility. Samples yaw uniformly from
    the given range, then samples X/Y within the region while accounting for
    the rotated block footprint. Z is placed at the support plane plus half-height.

    Returns:
        position: (x, y, z) in meters
        orientation: (w, x, y, z) quaternion
    """
    rng = random.Random(int(seed))

    # Sample yaw uniformly from range
    yaw_deg = rng.uniform(yaw_range_deg[0], yaw_range_deg[1])
    yaw_rad = math.radians(yaw_deg)

    # Calculate half-extents accounting for rotated footprint
    half_x = (
        abs(math.cos(yaw_rad)) * prop_width + abs(math.sin(yaw_rad)) * prop_depth
    ) / 2.0
    half_y = (
        abs(math.sin(yaw_rad)) * prop_width + abs(math.cos(yaw_rad)) * prop_depth
    ) / 2.0

    # Validate region can hold the prop
    if (
        region_min[0] + half_x > region_max[0] - half_x
        or region_min[1] + half_y > region_max[1] - half_y
    ):
        raise ValueError(
            f"region cannot hold prop at yaw {yaw_deg}; "
            f"need at least {half_x * 2}x{half_y * 2}m but region is "
            f"{region_max[0] - region_min[0]}x{region_max[1] - region_min[1]}m"
        )

    # Sample X/Y within valid bounds
    x = rng.uniform(region_min[0] + half_x, region_max[0] - half_x)
    y = rng.uniform(region_min[1] + half_y, region_max[1] - half_y)

    # Z is at support plane + half-height
    z = region_min[2] + prop_height / 2.0

    position = (x, y, z)
    orientation = _yaw_to_quaternion_wxyz(yaw_deg)

    return position, orientation


def _sample_gaussian_position(
    seed: int,
    stream: str,
    target: tuple[float, float, float],
    standard_deviation: float,
    *,
    axes: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[float, float, float]:
    rng = random.Random(f"{stream}:{int(seed)}")
    return (
        rng.gauss(target[0], standard_deviation) if axes[0] else target[0],
        rng.gauss(target[1], standard_deviation) if axes[1] else target[1],
        rng.gauss(target[2], standard_deviation) if axes[2] else target[2],
    )


def _sample_height_scaled_position(
    seed: int,
    stream: str,
    target: tuple[float, float, float],
    ratio: float,
    *,
    axes: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[float, float, float]:
    return _sample_gaussian_position(
        seed, stream, target, abs(target[2]) * ratio, axes=axes
    )


def sample_center_position(
    seed: int, target: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Sample a reproducible noisy wrist position around the centering target."""
    return _sample_height_scaled_position(
        seed, "center-arrival", target, CENTER_NOISE_STDEV_RATIO
    )


def grasp_height(seed: int) -> float:
    """Sample a reproducible grasp height within the configured range."""
    return random.Random(f"grasp-height:{int(seed)}").uniform(*GRASP_HEIGHT_RANGE_M)


@dataclass(frozen=True)
class PickConfig:
    prop_name: str = "pick_block"
    arm_name: str = "pick-arm"
    gripper_name: str = "pick-gripper"
    approach_clearance_m: float = 0.15
    grasp_height_m: float = PICK_BLOCK_HEIGHT_M / 2.0
    lift_height_m: float = 0.20
    minimum_lift_m: float = 0.10
    yaw_offset_deg: float = 0.0
    gripper_settle_sec: float = DEFAULT_GRIPPER_SETTLE_SEC
    grip_only: bool = False
    grip_lift_m: float = 0.05
    noise: bool = True
    pick_region_min_m: tuple[float, float, float] = PICK_REGION_MIN_ARM_M
    pick_region_max_m: tuple[float, float, float] = PICK_REGION_MAX_ARM_M
    prop_width_m: float = PICK_BLOCK_WIDTH_M
    prop_depth_m: float = PICK_BLOCK_DEPTH_M
    prop_height_m: float = PICK_BLOCK_HEIGHT_M
    # Optional second block to place alongside the target, and the target's
    # label (e.g. "red") used in stage texts.
    distractor_prop: str | None = None
    target_label: str = ""
    # Optional flat pad to put the block on after lifting it, and its label
    # (e.g. "green") used in stage texts.
    place_prop: str | None = None
    place_label: str = ""


@dataclass(frozen=True)
class EpisodeConfig:
    output_dir: str | None
    episode_id: str
    task: str = DEFAULT_PROMPT
    arm_name: str = "pick-arm"
    fixed_camera_name: str = "scene-cam"
    wrist_camera_name: str = "wrist-cam"
    fixed_camera_2_name: str = ""
    cosmos_output_dir: str | None = None
    upload: Mapping[str, Any] | None = None
    nudge: Mapping[str, Any] | None = None


def _episode_config(args: argparse.Namespace) -> EpisodeConfig | None:
    """Resolve native and optional CosmosWriter output from public CLI inputs."""
    run_root_arg = args.episode_output_dir
    if args.cosmos_output and not run_root_arg:
        raise ValueError("--cosmos-output requires --episode-output-dir")
    if not run_root_arg:
        return None

    episode_id = args.episode_id or f"{args.seed:06d}"
    run_root = Path(run_root_arg).expanduser().resolve()
    native_root = run_root / "episodes"
    cosmos_episode_dir = (
        run_root / "cosmoswriter" / episode_id if args.cosmos_output else None
    )
    return EpisodeConfig(
        output_dir=str(native_root),
        episode_id=episode_id,
        task=args.task,
        arm_name=args.arm,
        fixed_camera_name=args.fixed_camera,
        wrist_camera_name=args.wrist_camera,
        fixed_camera_2_name=args.fixed_camera_2,
        cosmos_output_dir=(
            str(cosmos_episode_dir) if cosmos_episode_dir is not None else None
        ),
    )


def _yaw_degrees(rotation_wxyz: list[float]) -> float:
    w, x, y, z = (float(value) for value in rotation_wxyz)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _nearest_symmetric_yaw_degrees(yaw_deg: float, period_deg: float = 180.0) -> float:
    """A parallel-jaw grasp is unchanged by 180 degrees, and a prop with a
    square footprint looks and grasps the same every 90; choose the short turn.

    Using the prop's full symmetry keeps the grasp yaw a function of what the
    cameras see, so identical-looking scenes get identical wrist labels."""
    half = period_deg / 2.0
    return (yaw_deg + half) % period_deg - half


async def resolve_pick_region(
    machine: Any, arm_name: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    corners = []
    for x_m in (PICK_REGION_MIN_ARM_M[0], PICK_REGION_MAX_ARM_M[0]):
        for y_m in (PICK_REGION_MIN_ARM_M[1], PICK_REGION_MAX_ARM_M[1]):
            transformed = await machine.transform_pose(
                PoseInFrame(
                    reference_frame=f"{arm_name}:base",
                    pose=Pose(
                        x=x_m * 1000.0,
                        y=y_m * 1000.0,
                        z=0.0,
                        o_z=1.0,
                    ),
                ),
                "world",
            )
            corners.append((transformed.pose.x / 1000.0, transformed.pose.y / 1000.0))
    xs, ys = zip(*corners)
    return (
        (min(xs), min(ys), 0.0),
        (max(xs), max(ys), 0.0),
    )


def _destination(
    x_m: float, y_m: float, z_m: float, world_yaw_deg: float
) -> PoseInFrame:
    # At the downward OV pole, Viam's ZYZ orientation maps positive theta to
    # negative world yaw. Negate it so the tool's horizontal axis follows the
    # block yaw rather than mirroring it across the world X axis.
    return PoseInFrame(
        reference_frame="world",
        pose=Pose(
            x=x_m * 1000.0,
            y=y_m * 1000.0,
            z=z_m * 1000.0,
            o_x=0.0,
            o_y=0.0,
            o_z=-1.0,
            theta=-world_yaw_deg,
        ),
    )


def _position_m(destination: PoseInFrame) -> tuple[float, float, float]:
    return (
        destination.pose.x / 1000.0,
        destination.pose.y / 1000.0,
        destination.pose.z / 1000.0,
    )


def _with_position(
    destination: PoseInFrame, position_m: tuple[float, float, float]
) -> PoseInFrame:
    pose = destination.pose
    return PoseInFrame(
        reference_frame=destination.reference_frame,
        pose=Pose(
            x=position_m[0] * 1000.0,
            y=position_m[1] * 1000.0,
            z=position_m[2] * 1000.0,
            o_x=pose.o_x,
            o_y=pose.o_y,
            o_z=pose.o_z,
            theta=pose.theta,
        ),
    )


def _midpoint(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        (start[0] + end[0]) / 2.0,
        (start[1] + end[1]) / 2.0,
        (start[2] + end[2]) / 2.0,
    )


async def _move(
    motion: Any,
    component_name: str,
    destination: PoseInFrame,
    *,
    constraints: Constraints | None = None,
) -> None:
    if not await motion.move(component_name, destination, constraints=constraints):
        raise RuntimeError(f"motion service could not move {component_name}")


def _finite_sim_time(status: Mapping[str, Any]) -> float | None:
    """The world's simulator clock, or None when it reports none (mock
    backend, real robot)."""
    raw = status.get("sim_time")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if not math.isfinite(raw):
        return None
    return float(raw)


class _ElapsedClock:
    """Elapsed seconds on the simulator clock when the world has one, else on
    the wall clock, so timeouts bound simulator time wherever it exists."""

    def __init__(self, world: Any, label: str, sim_start: float | None) -> None:
        self._world = world
        self._label = label
        self._sim_start = sim_start
        self._wall_start = time.monotonic()

    @classmethod
    async def start(cls, world: Any, label: str) -> _ElapsedClock:
        status = await world.do_command({"command": "status"})
        return cls(world, label, _finite_sim_time(status))

    @property
    def simulated(self) -> bool:
        return self._sim_start is not None

    async def elapsed(self, *, check_playing: bool = True) -> float:
        if self._sim_start is None:
            return time.monotonic() - self._wall_start
        status = await self._world.do_command({"command": "status"})
        now = _finite_sim_time(status)
        if now is None:
            raise RuntimeError("simulator status stopped reporting finite sim_time")
        if check_playing and status.get("playing") is False:
            raise RuntimeError(f"simulator paused during {self._label}")
        return now - self._sim_start


class _PhaseTimer:
    """Wall-clock seconds per named phase of one pick, reported in the result
    so run.json shows where episode time goes."""

    def __init__(self) -> None:
        self._started = time.monotonic()
        self._phases: dict[str, float] = {}
        self.moves: list[dict[str, Any]] = []
        self.waits: dict[str, Any] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.add(name, time.monotonic() - start)

    def add(self, name: str, seconds: float) -> None:
        self._phases[name] = self._phases.get(name, 0.0) + seconds

    def summary(self) -> dict[str, Any]:
        return {
            "wall_total_sec": time.monotonic() - self._started,
            "phases_sec": dict(self._phases),
            "moves": list(self.moves),
            "gripper_waits": dict(self.waits),
        }


async def _wait_sim_duration(world: Any, duration_sec: float) -> None:
    if not math.isfinite(duration_sec) or duration_sec < 0.0:
        raise ValueError("gripper settle duration must be finite and non-negative")
    if duration_sec == 0.0:
        return
    status = await world.do_command({"command": "status"})
    start = _finite_sim_time(status)
    if start is None:
        await asyncio.sleep(duration_sec)
        return

    if status.get("playing") is False:
        raise RuntimeError("simulator paused during gripper settle wait")

    target = start + duration_sec
    poll_interval = min(0.05, duration_sec)
    while True:
        await asyncio.sleep(poll_interval)
        status = await world.do_command({"command": "status"})
        current = _finite_sim_time(status)
        if current is None:
            raise RuntimeError("simulator status stopped reporting finite sim_time")
        if current >= target:
            return
        if status.get("playing") is False:
            raise RuntimeError("simulator paused during gripper settle wait")


async def _gripper_is_moving(gripper: Any) -> bool | None:
    """IsMoving, or None when the gripper does not implement it (real
    hardware), in which case callers fall back to fixed waits."""
    try:
        return bool(await gripper.is_moving())
    except (NotImplementedError, GRPCError):
        return None


async def _wait_for_gripper_settled(
    world: Any, gripper: Any, timeout_sec: float, label: str
) -> dict[str, Any]:
    """Wait until the jaw has moved and stopped, bounded by timeout_sec.

    The jaw must be seen moving before "stopped" counts, so a gripper whose
    IsMoving is a stub that always reports False, or that does not implement
    it, waits the full timeout exactly as the previous fixed sleep did. Returns
    timing metadata for the episode record."""
    if not math.isfinite(timeout_sec) or timeout_sec < 0.0:
        raise ValueError("gripper settle timeout must be finite and non-negative")
    clock = await _ElapsedClock.start(world, label)
    jaw_moved = False
    settled = False
    fallback = False
    while True:
        elapsed = await clock.elapsed()
        if elapsed >= timeout_sec:
            break
        moving = await _gripper_is_moving(gripper)
        if moving is None:
            fallback = True
            await _wait_sim_duration(world, timeout_sec - elapsed)
            elapsed = await clock.elapsed()
            break
        if moving:
            jaw_moved = True
        elif jaw_moved:
            settled = True
            break
        await asyncio.sleep(GRIPPER_SETTLE_POLL_SEC)
    return {
        "settled": settled,
        "elapsed_sec": elapsed,
        "timeout_sec": timeout_sec,
        "clock": "sim" if clock.simulated else "wall",
        "fallback": fallback,
    }


async def _firm_grasp(gripper: Any) -> tuple[bool, dict[str, Any]]:
    holding = await gripper.is_holding_something()
    meta = dict(holding.meta)
    firm_grasp = meta.get("firm_grasp")
    if not isinstance(firm_grasp, bool):
        raise RuntimeError("gripper holding metadata omitted boolean firm_grasp")
    return firm_grasp, meta


async def _wait_for_firm_grasp(
    world: Any, gripper: Any, timeout_sec: float
) -> tuple[bool, Mapping[str, Any]]:
    """Wait for a firm grasp, bounded by timeout_sec of simulator time.

    Ends early once the jaw has moved and stopped without latching (a closed-jaw
    miss) instead of idling until the timeout. A gripper without a usable
    IsMoving never reports motion, so it keeps the full timeout."""
    if not math.isfinite(timeout_sec) or timeout_sec < 0.0:
        raise ValueError("gripper close timeout must be finite and non-negative")

    clock = await _ElapsedClock.start(world, "gripper close wait")
    jaw_moved = False
    while True:
        firm_grasp, meta = await _firm_grasp(gripper)
        if firm_grasp:
            await _wait_sim_duration(world, FIRM_GRASP_SETTLE_SEC)
            return True, meta
        if await clock.elapsed() >= timeout_sec:
            return False, meta
        moving = await _gripper_is_moving(gripper)
        if moving:
            jaw_moved = True
        elif moving is False and jaw_moved:
            # The latch can flip between the two reads, so re-check holding
            # before calling the stop a miss.
            firm_grasp, meta = await _firm_grasp(gripper)
            if firm_grasp:
                await _wait_sim_duration(world, FIRM_GRASP_SETTLE_SEC)
            return firm_grasp, meta
        await asyncio.sleep(FIRM_GRASP_POLL_SEC)


async def _plan_move(
    motion: Any,
    component_name: str,
    destination: PoseInFrame,
    *,
    constraints: Constraints | None = None,
) -> list[Mapping[str, Any]]:
    request = MoveRequest(
        name=motion.name,
        component_name=component_name,
        destination=destination,
        constraints=constraints,
    )
    response = await motion.do_command({"plan": MessageToJson(request)})
    trajectory = response.get("plan")
    if (
        not isinstance(trajectory, list)
        or not trajectory
        or not all(isinstance(step, Mapping) for step in trajectory)
    ):
        raise RuntimeError(
            f"motion service returned an invalid plan for {component_name}"
        )
    return trajectory




def _trajectory_points(
    plan: list[Mapping[str, Any]],
    arm_name: str,
    max_velocity_rad_s: list[float],
    max_acceleration_rad_s2: list[float],
    minimum_duration_sec: float,
) -> list[Arm.TrajectoryPoint]:
    waypoints: list[list[float]] = []
    joint_count: int | None = None
    for index, step in enumerate(plan):
        raw_positions = step.get(arm_name)
        if not isinstance(raw_positions, list) or not raw_positions:
            raise RuntimeError(
                f"plan step {index} has no joint positions for {arm_name}"
            )
        positions = [float(value) for value in raw_positions]
        if not all(math.isfinite(value) for value in positions):
            raise RuntimeError(f"plan step {index} has non-finite positions")
        if joint_count is None:
            joint_count = len(positions)
        elif len(positions) != joint_count:
            raise RuntimeError("plan joint count changes between steps")
        if not waypoints or positions != waypoints[-1]:
            waypoints.append(positions)

    if len(waypoints) < 2:
        raise RuntimeError("plan needs at least two distinct joint waypoints")
    assert joint_count is not None

    def validate_limits(values: list[float], name: str) -> None:
        if len(values) != joint_count or not all(
            math.isfinite(value) and value > 0.0 for value in values
        ):
            raise RuntimeError(
                f"arm dynamics must provide {joint_count} finite positive {name}"
            )

    validate_limits(max_velocity_rad_s, "velocity limits")
    validate_limits(max_acceleration_rad_s2, "acceleration limits")
    if not math.isfinite(minimum_duration_sec) or minimum_duration_sec < 0.0:
        raise ValueError("minimum trajectory duration must be finite and non-negative")

    waypoint_array = np.asarray(waypoints, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(waypoint_array, axis=0), axis=1)
    path_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    path_positions /= path_positions[-1]
    path = ta.SplineInterpolator(path_positions, waypoint_array)
    velocity_limits = np.column_stack(
        (-np.asarray(max_velocity_rad_s), np.asarray(max_velocity_rad_s))
    )
    acceleration_limits = np.column_stack(
        (-np.asarray(max_acceleration_rad_s2), np.asarray(max_acceleration_rad_s2))
    )
    instance = ta_algorithm.TOPPRA(
        [
            ta_constraint.JointVelocityConstraint(velocity_limits),
            ta_constraint.JointAccelerationConstraint(acceleration_limits),
        ],
        path,
        solver_wrapper="seidel",
    )
    parameterized = instance.compute_trajectory(0.0, 0.0)
    if (
        parameterized is None
        or not math.isfinite(parameterized.duration)
        or parameterized.duration <= 0.0
    ):
        raise RuntimeError("TOPP-RA could not parameterize the plan")

    source_duration = float(parameterized.duration)
    duration = max(source_duration, minimum_duration_sec)
    command_period = 1.0 / TRAJECTORY_COMMAND_HZ
    times = np.arange(0.0, duration, command_period)
    if not math.isclose(float(times[-1]), duration):
        times = np.append(times, duration)
    source_times = times * (source_duration / duration)
    positions = parameterized(source_times, 0)
    velocities = parameterized(source_times, 1) * (source_duration / duration)
    accelerations = parameterized(source_times, 2) * (
        source_duration / duration
    ) ** 2
    velocities[0] = 0.0
    velocities[-1] = 0.0

    points = [
        Arm.TrajectoryPoint(
            time=timedelta(seconds=float(point_time)),
            positions=[math.degrees(float(value)) for value in point_positions],
            constraints=Arm.KinematicConstraints(
                velocities=[math.degrees(float(value)) for value in point_velocities],
                accelerations=[
                    math.degrees(float(value)) for value in point_accelerations
                ],
            ),
        )
        for point_time, point_positions, point_velocities, point_accelerations in zip(
            times, positions, velocities, accelerations
        )
    ]
    # numerical precision
    assert (points[-1].time.total_seconds() - minimum_duration_sec) > -0.001
    return points


async def _prepare_trajectory(
    arm: Any,
    arm_name: str,
    plan: list[Mapping[str, Any]],
    minimum_duration_sec: float,
) -> list[Arm.TrajectoryPoint]:
    metadata = await arm.do_command({"command": "dynamics_metadata"})
    raw_velocity_limits = metadata.get("max_velocity_rad_s")
    raw_acceleration_limits = metadata.get("max_acceleration_rad_s2")
    if not isinstance(raw_velocity_limits, list):
        raise RuntimeError("arm dynamics did not provide max_velocity_rad_s")
    if raw_acceleration_limits is None:
        acceleration_limits = [
            math.radians(DEFAULT_MAX_JOINT_ACCELERATION_DEG_S2)
        ] * len(raw_velocity_limits)
    elif isinstance(raw_acceleration_limits, list):
        acceleration_limits = [float(value) for value in raw_acceleration_limits]
    else:
        raise RuntimeError("arm dynamics provided invalid max_acceleration_rad_s2")
    return _trajectory_points(
        plan,
        arm_name,
        [float(value) for value in raw_velocity_limits],
        acceleration_limits,
        minimum_duration_sec,
    )


async def _execute_trajectory(
    arm: Any, points: list[Arm.TrajectoryPoint]
) -> None:
    async def batches():
        yield points

    received_update = False
    async for _ in arm.move_through_joint_positions_streamed(batches()):
        received_update = True
    if not received_update:
        raise RuntimeError("arm returned no update for the trajectory")


def _sample_clear_pose(
    seed: int,
    stream: str,
    config: PickConfig,
    target_position: Sequence[float],
    min_separation_m: float,
    size_m: tuple[float, float, float],
    yaw_range_deg: tuple[float, float],
    name: str | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    for attempt in range(DISTRACTOR_PLACEMENT_TRIES):
        position, orientation = sample_prop_pose(
            seed=random.Random(f"{stream}:{int(seed)}:{attempt}").randrange(2**31),
            region_min=config.pick_region_min_m,
            region_max=config.pick_region_max_m,
            yaw_range_deg=yaw_range_deg,
            prop_width=size_m[0],
            prop_depth=size_m[1],
            prop_height=size_m[2],
        )
        if math.dist(position[:2], target_position[:2]) >= min_separation_m:
            return position, orientation
    raise RuntimeError(
        f"could not place {name} at least {min_separation_m} m from the target"
    )


def sample_distractor_pose(
    seed: int, config: PickConfig, target_position: Sequence[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Sample a reproducible pose for the second block, clear of the target."""
    return _sample_clear_pose(
        seed,
        "distractor",
        config,
        target_position,
        DISTRACTOR_MIN_SEPARATION_M,
        (config.prop_width_m, config.prop_depth_m, config.prop_height_m),
        PICK_YAW_RANGE_DEG,
        config.distractor_prop,
    )


def sample_place_pose(
    seed: int, config: PickConfig, target_position: Sequence[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Sample a reproducible, axis-aligned pose for the place pad, clear of the block."""
    return _sample_clear_pose(
        seed,
        "place-pad",
        config,
        target_position,
        PLACE_MIN_SEPARATION_M,
        (PLACE_PAD_WIDTH_M, PLACE_PAD_WIDTH_M, PLACE_PAD_HEIGHT_M),
        (0.0, 0.0),
        config.place_prop,
    )


def block_on_pad(
    block_position: Sequence[float],
    pad_position: Sequence[float],
    block_height_m: float,
) -> bool:
    """True when the block's center is over the pad and it rests flat on it."""
    half = PLACE_PAD_WIDTH_M / 2.0
    rest_z = pad_position[2] + PLACE_PAD_HEIGHT_M / 2.0 + block_height_m / 2.0
    return (
        abs(block_position[0] - pad_position[0]) <= half
        and abs(block_position[1] - pad_position[1]) <= half
        and abs(block_position[2] - rest_z) <= PLACE_REST_TOLERANCE_M
    )


async def execute_pick(
    world: Any,
    gripper: Any,
    motion: Any,
    arm: Any,
    seed: int,
    config: PickConfig = PickConfig(),
    episode: EpisodeConfig | None = None,
) -> Mapping[str, Any]:
    """Reset the target, grasp it from above, verify lift, and finish while holding."""
    if config.grip_only and config.grip_lift_m <= 0:
        raise ValueError("grip_lift_m must be positive in grip-only mode")

    timer = _PhaseTimer()
    sim_clock = await _ElapsedClock.start(world, "episode")
    position, orientation = sample_prop_pose(
        seed=seed,
        region_min=config.pick_region_min_m,
        region_max=config.pick_region_max_m,
        yaw_range_deg=PICK_YAW_RANGE_DEG,
        prop_width=config.prop_width_m,
        prop_depth=config.prop_depth_m,
        prop_height=config.prop_height_m,
    )
    with timer.phase("prop_reset"):
        result = await world.do_command(
            {
                "command": "set_prop_pose",
                "prop": config.prop_name,
                "position": list(position),
                "orientation_wxyz": list(orientation),
            }
        )
    target = {
        "position": result["position"],
        "orientation_wxyz": result["orientation_wxyz"],
    }
    if config.distractor_prop:
        distractor_position, distractor_orientation = sample_distractor_pose(
            seed, config, target["position"]
        )
        await world.do_command(
            {
                "command": "set_prop_pose",
                "prop": config.distractor_prop,
                "position": list(distractor_position),
                "orientation_wxyz": list(distractor_orientation),
            }
        )
    pad_position: tuple[float, float, float] | None = None
    if config.place_prop:
        pad_position, pad_orientation = sample_place_pose(
            seed, config, target["position"]
        )
        await world.do_command(
            {
                "command": "set_prop_pose",
                "prop": config.place_prop,
                "position": list(pad_position),
                "orientation_wxyz": list(pad_orientation),
            }
        )
    stages = stage_prompts(config.target_label, config.place_label)
    x_m, y_m, initial_z_m = (float(value) for value in target["position"])
    yaw_deg = _nearest_symmetric_yaw_degrees(
        _yaw_degrees(target["orientation_wxyz"]) + config.yaw_offset_deg,
        90.0 if math.isclose(config.prop_width_m, config.prop_depth_m) else 180.0,
    )
    grasp_height_m = grasp_height(seed) if config.noise else config.grasp_height_m
    exact_center_position = (x_m, y_m, initial_z_m + config.approach_clearance_m)
    center_position = (
        sample_center_position(seed, exact_center_position)
        if config.noise
        else exact_center_position
    )
    center = _destination(*center_position, 0.0)
    approach = _destination(
        x_m, y_m, initial_z_m + config.approach_clearance_m, yaw_deg
    )
    grasp = _destination(x_m, y_m, grasp_height_m, yaw_deg)
    exact_lift_position = (x_m, y_m, initial_z_m + config.lift_height_m)
    lift_position = (
        _sample_height_scaled_position(
            seed, "lift", exact_lift_position, LIFT_NOISE_STDEV_RATIO
        )
        if config.noise
        else exact_lift_position
    )
    lift = _destination(*lift_position, yaw_deg)
    pre_grasp = _destination(
        x_m,
        y_m,
        grasp_height_m + PRE_GRASP_CLEARANCE_M,
        yaw_deg,
    )
    grip_lift = _destination(x_m, y_m, grasp_height_m + config.grip_lift_m, yaw_deg)

    center_waypoints = [center]
    descent_waypoints = [
        (pre_grasp, MIN_SUBTASK_DURATION_SEC, False),
        (grasp, FINAL_DESCENT_MIN_DURATION_SEC, True),
    ]
    if config.noise and not config.grip_only:
        start = await motion.get_pose(config.gripper_name, "world")
        start_position = _sample_height_scaled_position(
            seed, "start", _position_m(start), START_NOISE_STDEV_RATIO
        )
        noisy_start = _with_position(start, start_position)
        with timer.phase("start_move"):
            await _move(
                motion,
                config.gripper_name,
                noisy_start,
                constraints=CENTER_MOVE_CONSTRAINTS,
            )

        center_midpoint = _sample_height_scaled_position(
            seed,
            "center-midpoint",
            _midpoint(start_position, center_position),
            CENTER_MIDPOINT_NOISE_STDEV_RATIO,
        )
        center_waypoints = [_destination(*center_midpoint, 0.0), center]

        exact_descent_midpoint = _midpoint(center_position, _position_m(pre_grasp))
        descent_midpoint = _sample_height_scaled_position(
            seed,
            "descent-midpoint",
            exact_descent_midpoint,
            DESCENT_MIDPOINT_NOISE_STDEV_RATIO,
            axes=(True, True, False),
        )
        # Correct lateral and rotational noise at the clearance plane. Only the
        # final waypoint descends between the open jaws to grasp height.
        descent_arrival_position = _sample_gaussian_position(
            seed,
            "descent-arrival-position",
            _position_m(pre_grasp),
            DESCENT_ARRIVAL_POSITION_STDEV_M,
            axes=(True, True, False),
        )
        descent_arrival_yaw = random.Random(
            f"descent-arrival-rotation:{int(seed)}"
        ).gauss(yaw_deg, DESCENT_ARRIVAL_ROTATION_STDEV_DEG)
        descent_waypoints = [
            (
                _destination(*descent_midpoint, yaw_deg),
                APPROACH_WAYPOINT_MIN_DURATION_SEC,
                False,
            ),
            (
                _destination(*descent_arrival_position, descent_arrival_yaw),
                APPROACH_WAYPOINT_MIN_DURATION_SEC,
                False,
            ),
            (
                _destination(*descent_arrival_position, yaw_deg),
                CORRECTION_MOVE_MIN_DURATION_SEC,
                False,
            ),
            (pre_grasp, CORRECTION_MOVE_MIN_DURATION_SEC, False),
            (grasp, FINAL_DESCENT_MIN_DURATION_SEC, True),
        ]

    center_trajectory = None
    center_plan_sec = 0.0
    descent_trajectories: list[list[Arm.TrajectoryPoint]] = []
    if not config.grip_only:
        with timer.phase("plan"):
            plan_started = time.monotonic()
            center_plan = await _plan_move(
                motion,
                config.gripper_name,
                center_waypoints[0],
                constraints=CENTER_MOVE_CONSTRAINTS,
            )
            center_trajectory = await _prepare_trajectory(
                arm,
                config.arm_name,
                center_plan,
                MIN_SUBTASK_DURATION_SEC / len(center_waypoints),
            )
            center_plan_sec = time.monotonic() - plan_started

    with timer.phase("gripper_open"):
        await gripper.open()
        # Opening never contacts anything, so a short bound is enough outside
        # grip-only mode; the wait ends as soon as the jaw stops either way.
        timer.waits["open"] = await _wait_for_gripper_settled(
            world,
            gripper,
            config.gripper_settle_sec if config.grip_only else OPEN_SETTLE_MAX_SEC,
            "gripper open wait",
        )
    if config.grip_only:
        with timer.phase("pre_grasp_move"):
            await _move(motion, config.gripper_name, pre_grasp)

    capture_started = False
    if episode is not None:
        start_command: dict[str, Any] = {
            "command": "start_episode",
            "output_dir": episode.output_dir,
            "episode_id": episode.episode_id,
            "seed": seed,
            "task": episode.task,
            "arm": episode.arm_name,
            "gripper": config.gripper_name,
            "fixed_camera": episode.fixed_camera_name,
            "fixed_camera_2": episode.fixed_camera_2_name,
            "wrist_camera": episode.wrist_camera_name,
            "cosmos_output_dir": episode.cosmos_output_dir,
            "target": {
                "prop_name": config.prop_name,
                "position_m": target["position"],
                "rotation_wxyz": target["orientation_wxyz"],
            },
        }
        if episode.nudge:
            start_command["nudge"] = dict(episode.nudge)
        with timer.phase("capture_start"):
            await world.do_command(start_command)
        capture_started = True

    async def mark_stage(task: str) -> None:
        if capture_started:
            await world.do_command({"command": "mark_episode_stage", "task": task})

    async def prepare_stage_trajectory(
        destination: PoseInFrame,
        constraints: Constraints,
        minimum_duration_sec: float,
        *,
        start_stage: str | None = None,
    ) -> tuple[list[Arm.TrajectoryPoint], float]:
        """Plan with physics paused; returns the trajectory and the wall
        seconds spent planning."""
        plan_started = time.monotonic()
        if capture_started:
            await world.do_command({"command": "pause"})
        try:
            plan = await _plan_move(
                motion,
                config.gripper_name,
                destination,
                constraints=constraints,
            )
            trajectory = await _prepare_trajectory(
                arm,
                config.arm_name,
                plan,
                minimum_duration_sec,
            )
            if start_stage is not None:
                await mark_stage(start_stage)
        finally:
            if capture_started:
                await world.do_command({"command": "play"})
        plan_sec = time.monotonic() - plan_started
        timer.add("plan", plan_sec)
        return trajectory, plan_sec

    async def run_trajectory(
        label: str, trajectory: list[Arm.TrajectoryPoint], plan_sec: float
    ) -> None:
        """Stream one trajectory and record wall time against its planned
        duration, which is the idle-waiting signal for this move."""
        execute_started = time.monotonic()
        await _execute_trajectory(arm, trajectory)
        execute_sec = time.monotonic() - execute_started
        timer.add("execute", execute_sec)
        timer.moves.append(
            {
                "label": label,
                "plan_sec": plan_sec,
                "execute_sec": execute_sec,
                "planned_sec": trajectory[-1].time.total_seconds(),
            }
        )

    outcome = "incomplete"
    capture = None
    try:
        if not config.grip_only:
            # todo: construct the full prompt from these subprompts.
            await mark_stage(stages[0])
            assert center_trajectory is not None
            await run_trajectory("center", center_trajectory, center_plan_sec)
            for index, destination in enumerate(center_waypoints[1:], start=1):
                trajectory, plan_sec = await prepare_stage_trajectory(
                    destination,
                    CENTER_MOVE_CONSTRAINTS,
                    MIN_SUBTASK_DURATION_SEC / len(center_waypoints),
                )
                await run_trajectory(f"center-{index}", trajectory, plan_sec)

            await mark_stage(stages[1])
            for index, (destination, minimum_duration, starts_grasp) in enumerate(
                descent_waypoints
            ):
                trajectory, plan_sec = await prepare_stage_trajectory(
                    destination,
                    LINEAR_PICK_CONSTRAINTS,
                    minimum_duration,
                    start_stage=stages[2] if starts_grasp else None,
                )
                descent_trajectories.append(trajectory)
                await run_trajectory(f"descent-{index}", trajectory, plan_sec)

        else:
            # note: we don't really care about stages for grip-only mode, grip-only mode is for debugging grip issues.
            await mark_stage("Lower the gripper to grasp the block")
            with timer.phase("grasp_move"):
                await _move(
                    motion,
                    config.gripper_name,
                    grasp,
                    constraints=LINEAR_PICK_CONSTRAINTS,
                )
            await mark_stage("Close the gripper and lift the block")
        with timer.phase("grab"):
            grab_commanded = bool(await gripper.grab())
        holding_meta: Mapping[str, Any] = {}
        if grab_commanded:
            with timer.phase("grasp_settle"):
                grabbed, holding_meta = await _wait_for_firm_grasp(
                    world, gripper, config.gripper_settle_sec
                )
        else:
            grabbed = False
        holding = False
        lifted_z_m = initial_z_m

        if config.grip_only:
            with timer.phase("lift_move"):
                await _move(
                    motion,
                    config.gripper_name,
                    grip_lift,
                    constraints=LINEAR_PICK_CONSTRAINTS,
                )
        else:
            assert descent_trajectories
            ascent_destination = lift if grabbed else approach
            descent_duration = sum(
                trajectory[-1].time.total_seconds()
                for trajectory in descent_trajectories
            )
            ascent_minimum = max(
                # this could be faster, we actually want this one slow
                # 0.0, MIN_SUBTASK_DURATION_SEC - descent_duration
                0.0, MIN_SUBTASK_DURATION_SEC
            )
            ascent_trajectory, plan_sec = await prepare_stage_trajectory(
                ascent_destination,
                LINEAR_PICK_CONSTRAINTS,
                ascent_minimum,
            )
            assert (
                descent_duration + ascent_trajectory[-1].time.total_seconds()
                >= MIN_SUBTASK_DURATION_SEC
            )
            await run_trajectory(
                "lift" if grabbed else "retreat", ascent_trajectory, plan_sec
            )

        if grabbed:
            with timer.phase("lift_check"):
                status = await gripper.is_holding_something()
                holding = bool(status.is_holding_something)
                holding_meta = dict(status.meta)
                measured = await world.do_command(
                    {"command": "get_prop_pose", "prop": config.prop_name}
                )
                lifted_z_m = float(measured["pose"]["position"][2])

        minimum_lift_m = config.minimum_lift_m
        if config.grip_only:
            minimum_lift_m = max(0.0, config.grip_lift_m - GRIP_ONLY_LIFT_TOLERANCE_M)
        success = grabbed and holding and lifted_z_m >= initial_z_m + minimum_lift_m
        if success and pad_position is not None and not config.grip_only:
            # TCP height above the block's bottom face, measured after the lift
            # so any slip in the grasp is accounted for.
            tcp_above_bottom_m = lift_position[2] - (
                lifted_z_m - config.prop_height_m / 2.0
            )
            release_z_m = (
                pad_position[2]
                + PLACE_PAD_HEIGHT_M / 2.0
                + tcp_above_bottom_m
                + PLACE_RELEASE_CLEARANCE_M
            )
            pad_x_m, pad_y_m = pad_position[0], pad_position[1]
            carry = _destination(pad_x_m, pad_y_m, lift_position[2], yaw_deg)
            pre_release = _destination(
                pad_x_m, pad_y_m, release_z_m + PRE_GRASP_CLEARANCE_M, yaw_deg
            )
            release = _destination(pad_x_m, pad_y_m, release_z_m, yaw_deg)
            retreat = _destination(
                pad_x_m, pad_y_m, release_z_m + config.approach_clearance_m, yaw_deg
            )
            place_moves = [
                (carry, CENTER_MOVE_CONSTRAINTS, MIN_SUBTASK_DURATION_SEC, stages[3]),
                (
                    pre_release,
                    LINEAR_PICK_CONSTRAINTS,
                    APPROACH_WAYPOINT_MIN_DURATION_SEC,
                    stages[4],
                ),
                (release, LINEAR_PICK_CONSTRAINTS, FINAL_DESCENT_MIN_DURATION_SEC, None),
            ]
            for index, (destination, constraints, minimum_duration, stage) in enumerate(
                place_moves
            ):
                trajectory, plan_sec = await prepare_stage_trajectory(
                    destination, constraints, minimum_duration, start_stage=stage
                )
                await run_trajectory(f"place-{index}", trajectory, plan_sec)
            with timer.phase("release"):
                await gripper.open()
                timer.waits["release"] = await _wait_for_gripper_settled(
                    world, gripper, OPEN_SETTLE_MAX_SEC, "gripper release wait"
                )
                # Let the block drop the release clearance and settle on the pad.
                await _wait_sim_duration(world, PLACE_RELEASE_SETTLE_SEC)
            trajectory, plan_sec = await prepare_stage_trajectory(
                retreat,
                LINEAR_PICK_CONSTRAINTS,
                MIN_SUBTASK_DURATION_SEC,
                start_stage=stages[5],
            )
            await run_trajectory("place-retreat", trajectory, plan_sec)
            with timer.phase("place_check"):
                status = await gripper.is_holding_something()
                holding = bool(status.is_holding_something)
                measured = await world.do_command(
                    {"command": "get_prop_pose", "prop": config.prop_name}
                )
                placed_position = [float(v) for v in measured["pose"]["position"]]
            success = not holding and block_on_pad(
                placed_position, pad_position, config.prop_height_m
            )
        outcome = "success" if success else "failure"
        if success and not config.grip_only:
            await mark_stage(stages[-1])

    finally:
        if capture_started:
            stop_command: dict[str, Any] = {
                "command": "stop_episode",
                "outcome": outcome,
            }
            if episode is not None and episode.upload is not None:
                stop_command["upload"] = dict(episode.upload)
            with timer.phase("capture_stop"):
                capture = await world.do_command(stop_command)

    timing = timer.summary()
    timing["sim_elapsed_sec"] = (
        await sim_clock.elapsed(check_playing=False) if sim_clock.simulated else None
    )
    return {
        "outcome": outcome,
        "seed": int(seed),
        "grasp_height_m": grasp_height_m,
        "target_pose": target,
        "grabbed": grabbed,
        "holding_after_lift": holding,
        "lifted_z_m": lifted_z_m,
        "holding_meta": holding_meta,
        "capture": capture,
        "timing": timing,
    }


async def _main(args: argparse.Namespace) -> Mapping[str, Any]:
    episode = _episode_config(args)
    api_key = os.environ.get("VIAM_API_KEY")
    api_key_id = os.environ.get("VIAM_API_KEY_ID")
    if not api_key or not api_key_id:
        raise ValueError("VIAM_API_KEY and _ID required in env")
    options = (
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
        if api_key and api_key_id
        else RobotClient.Options(dial_options=DialOptions(insecure=True))
    )
    if args.disable_webrtc:
        assert options.dial_options is not None
        options.dial_options.disable_webrtc = True
    async with await RobotClient.at_address(args.address, options) as machine:
        pick_region_min_m, pick_region_max_m = await resolve_pick_region(
            machine, args.arm
        )
        world = Generic.from_robot(machine, args.world)
        gripper = Gripper.from_robot(machine, args.gripper)
        motion = MotionClient.from_robot(machine, args.motion)
        arm = Arm.from_robot(machine, args.arm)
        await arm.move_to_joint_positions(
            JointPositions(values=list(HOME_JOINT_DEGREES))
        )
        return await execute_pick(
            world,
            gripper,
            motion,
            arm,
            args.seed,
            PickConfig(
                prop_name=args.prop,
                gripper_name=args.gripper,
                yaw_offset_deg=args.yaw_offset_deg,
                arm_name=args.arm,
                gripper_settle_sec=args.gripper_settle_sec,
                noise=not args.no_noise,
                pick_region_min_m=pick_region_min_m,
                pick_region_max_m=pick_region_max_m,
            ),
            episode,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="machine address from the Viam CONNECT tab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default=DEFAULT_PROMPT)
    parser.add_argument("--world", default="sim-world")
    parser.add_argument("--gripper", default="pick-gripper")
    parser.add_argument("--motion", default="builtin")
    parser.add_argument("--prop", default="pick_block")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--fixed-camera", default="scene-cam")
    parser.add_argument(
        "--fixed-camera-2",
        default="",
        help="optional second fixed camera component; empty disables it",
    )
    parser.add_argument("--wrist-camera", default="wrist-cam")
    parser.add_argument(
        "--disable-webrtc",
        action="store_true",
        help="dial directly; needed for an authenticated localhost viam-server",
    )
    parser.add_argument(
        "--episode-output-dir",
        help="run root; write synchronized native data under RUN_ROOT/episodes",
    )
    parser.add_argument(
        "--cosmos-output",
        action="store_true",
        help=(
            "also write CosmosWriter output under RUN_ROOT/cosmoswriter/EPISODE_ID "
            "(requires --episode-output-dir)"
        ),
    )
    parser.add_argument("--episode-id", help="defaults to the zero-padded seed")
    parser.add_argument(
        "--gripper-settle-sec",
        type=float,
        default=PickConfig.gripper_settle_sec,
        help="maximum simulator seconds to wait for a firm grasp",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="alignment offset between the block long axis and gripper jaws",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="disable all action noise and use the configured grasp height",
    )
    args = parser.parse_args()
    if args.gripper_settle_sec < 0:
        parser.error("--gripper-settle-sec must be non-negative")
    result = asyncio.run(_main(args))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
