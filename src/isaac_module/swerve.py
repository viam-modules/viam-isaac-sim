"""Wheel-level swerve bases: each module steers its wheel to its own share of
the body velocity and rolls it there.

Ported from viam-mujoco-sim's swerve drive (bases.swerve_targets and the base
handle's control loop), with the same Viam semantics. Everything is in the
base's Viam frame: +y forward, +x right, yaw positive to the left; metres,
radians, seconds. Pure Python except stage_modules (pxr, sim thread).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

FORWARD_AXES = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
STILL_MPS = 1e-4  # a module slower than this keeps its steer angle
HEADING_GAIN = 3.0  # 1/s: yaw rate per radian off the held heading
DEFAULT_ACCEL_MPS2 = 1.0  # MoveStraight/Spin braking ramp
MIN_MPS = 0.02  # the ramp's least speed (or the command, if slower)
MIN_RAD_S = math.radians(2)
PROGRESS_M = 1e-3  # less progress than this (or its angle) counts as none
DEFAULT_STALL_S = 3.0


@dataclass(frozen=True)
class SwerveModule:
    steer: str  # steering joint (vertical axis)
    drive: str  # the wheel's joint, below it
    position: tuple[float, float]  # the steer axis in the Viam frame, m
    roll0: float  # way a positive drive speed rolls the wheel at steer 0 (Viam frame angle, rad)
    steer_sign: float  # +1 when a positive steer angle turns the wheel left


def to_viam_xy(xy: Sequence[float], forward_axis: str) -> tuple[float, float]:
    """A vector in the robot file's frame (its forward_axis is forward, z up)
    in the Viam frame (+y forward, +x right)."""
    fx, fy = FORWARD_AXES[forward_axis]
    x, y = float(xy[0]), float(xy[1])
    # forward component along f, left component along z x f = (-fy, fx)
    forward, left = x * fx + y * fy, -x * fy + y * fx
    return (-left, forward)


def parse_swerve_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Check a swerve base's attributes; returns pairs, wheel_radius_m,
    forward_axis, chassis_link."""
    modules = attrs.get("swerve_modules")
    if not isinstance(modules, Sequence) or isinstance(modules, str) or len(modules) < 2:
        raise ValueError(
            'swerve_modules must list at least two {"steer_joint", "drive_joint"} objects'
        )
    pairs = []
    for module in modules:
        if not isinstance(module, Mapping) or not all(
            isinstance(module.get(k), str) and module.get(k) for k in ("steer_joint", "drive_joint")
        ):
            raise ValueError(f"swerve_modules: {module!r} needs steer_joint and drive_joint names")
        pairs.append((module["steer_joint"], module["drive_joint"]))
    try:
        radius = float(attrs.get("wheel_radius_m"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("swerve_modules needs wheel_radius_m (the wheels' radius in metres)") from None
    if not (math.isfinite(radius) and radius > 0):
        raise ValueError("wheel_radius_m must be a positive number")
    forward_axis = attrs.get("forward_axis") or "+x"
    if forward_axis not in FORWARD_AXES:
        raise ValueError(f"forward_axis must be one of {sorted(FORWARD_AXES)}, not {forward_axis!r}")
    chassis = attrs.get("chassis_link", "base_link")
    if not isinstance(chassis, str) or not chassis:
        raise ValueError("chassis_link must name the link the modules are fixed to")
    return {"pairs": pairs, "wheel_radius_m": radius, "forward_axis": forward_axis, "chassis_link": chassis}


def swerve_targets(
    modules: Sequence[SwerveModule], u: float, v: float, w: float, previous: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Steer angles (rad, the joints' own, unwrapped) and wheel surface speeds
    (m/s) for the body velocity (u right, v forward, m/s; w left, rad/s). Each
    module turns to its own velocity, v + w x position, by whichever of the
    two wheel directions is the shorter turn from `previous` (reversing the
    wheel for the other); a module that would barely move keeps its angle."""
    angles, speeds = [float(a) for a in previous], [0.0] * len(modules)
    for i, m in enumerate(modules):
        vx, vy = u - w * m.position[1], v + w * m.position[0]
        speed = math.hypot(vx, vy)
        if speed < STILL_MPS:
            continue
        target = m.steer_sign * (math.atan2(vy, vx) - m.roll0)
        # the nearest of target + k*pi to where the joint is; odd k rolls the wheel backward
        k = round((angles[i] - target) / math.pi)
        angles[i], speeds[i] = target + k * math.pi, speed * (-1.0 if k % 2 else 1.0)
    return angles, speeds


@dataclass
class Goal:
    kind: str  # "straight" (metres along the start heading) or "spin" (radians)
    target: float  # signed
    rate: float  # signed m/s or rad/s
    done: threading.Event
    error: BaseException | None = None
    start: tuple[float, float, float] | None = None  # pose on the first step
    best: float = math.inf  # least remaining so far
    best_t: float = 0.0


class SwerveDrive:
    """The swerve controller. Not thread safe: the caller serializes it. step()
    runs every physics step with the chassis pose (x, y world metres; yaw the
    heading of the base's forward, radians, unwrapped by the caller or not)."""

    def __init__(
        self,
        modules: Sequence[SwerveModule],
        max_linear_mps: float,
        max_angular_rps: float,
        accel_mps2: float = DEFAULT_ACCEL_MPS2,
        stall_s: float = DEFAULT_STALL_S,
    ) -> None:
        self.modules = list(modules)
        self.max_linear = float(max_linear_mps)
        self.max_angular = float(max_angular_rps)
        self.accel = float(accel_mps2)
        self.angular_accel = self.accel / max(max(math.hypot(*m.position) for m in self.modules), 1e-3)
        self.stall_s = float(stall_s)
        self.angles = [0.0] * len(self.modules)  # commanded, kept: steering to measured noise chatters
        self.command = (0.0, 0.0, 0.0)  # u right, v forward, w left
        self.goal: Goal | None = None
        self._hold: float | None = None
        self._yaw: float | None = None  # unwrapped

    @property
    def moving(self) -> bool:
        return self.goal is not None or self.command != (0.0, 0.0, 0.0)

    def _end_goal(self, error: BaseException | None) -> None:
        goal, self.goal = self.goal, None
        if goal is not None:
            goal.error = error
            goal.done.set()

    def set_velocity(self, u: float, v: float, w: float) -> None:
        """Latch a body velocity, the linear part clamped as one vector."""
        if not all(math.isfinite(x) for x in (u, v, w)):
            raise ValueError("velocity must be finite")
        self._end_goal(RuntimeError("move was superseded by a new command"))
        scale = min(1.0, self.max_linear / max(math.hypot(u, v), 1e-12))
        w = max(-self.max_angular, min(self.max_angular, w))
        self.command = (u * scale, v * scale, w)

    def stop(self) -> None:
        self._end_goal(RuntimeError("move was stopped"))
        self.command = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        """Forget the motion state after a world reset teleports the base."""
        self._end_goal(RuntimeError("move was interrupted by a world reset"))
        self.command = (0.0, 0.0, 0.0)
        self._hold = self._yaw = None

    def start_goal(self, kind: str, target: float, rate: float) -> Goal:
        """MoveStraight (metres) or Spin (radians); the sign of target is the
        direction, rate's magnitude the speed (capped at the maximum)."""
        self._end_goal(RuntimeError("move was superseded by a new command"))
        self.command = (0.0, 0.0, 0.0)
        top = self.max_linear if kind == "straight" else self.max_angular
        goal = Goal(kind, float(target), math.copysign(min(abs(rate), top), target), threading.Event())
        if target == 0.0:
            goal.done.set()
            return goal
        goal.best = abs(goal.target)
        self.goal = goal
        return goal

    def step(
        self,
        x: float,
        y: float,
        yaw: float,
        dt: float,
        now: float,
        measured: Sequence[float] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Steer angle targets (rad) and wheel surface speeds (m/s)."""
        self._yaw = yaw if self._yaw is None else self._yaw + math.remainder(yaw - self._yaw, math.tau)
        yaw = self._yaw
        u, v, w = self.command
        goal = self.goal
        if goal is not None:
            u, v, w = self._goal_command(goal, x, y, yaw, now)
            self._hold = None
        elif w == 0.0 and (u or v):
            # moving without turning holds the heading it set off on: modules
            # turning to a new direction scrub, and would twist the base
            if self._hold is None:
                self._hold = yaw
            w = HEADING_GAIN * (self._hold - yaw)
        else:
            self._hold = None
        angles, speeds = swerve_targets(self.modules, u, v, w, self.angles)
        self.angles = angles
        if measured is not None:
            # a module still turning drives only its aligned share, so the wheels
            # don't fight while they steer (a real swerve's cosine scaling)
            speeds = [s * max(0.0, math.cos(m - a)) for s, m, a in zip(speeds, measured, angles)]
        return angles, speeds

    def _goal_command(self, goal: Goal, x: float, y: float, yaw: float, now: float) -> tuple[float, float, float]:
        if goal.start is None:
            goal.start, goal.best_t = (x, y, yaw), now
        x0, y0, yaw0 = goal.start
        if goal.kind == "spin":
            progress = yaw - yaw0
        else:
            progress = (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)
        remaining = abs(goal.target) - progress * math.copysign(1.0, goal.target)
        if remaining <= 0.0:
            self._end_goal(None)
            return 0.0, 0.0, 0.0
        spin = goal.kind == "spin"
        if remaining < goal.best - (math.radians(0.1) if spin else PROGRESS_M):
            goal.best, goal.best_t = remaining, now
        elif now - goal.best_t >= self.stall_s:
            done = math.degrees(progress) if spin else 1000 * progress
            wanted = math.degrees(goal.target) if spin else 1000 * goal.target
            self._end_goal(RuntimeError(
                f"base stalled: moved {done:.0f} of {wanted:.0f} {'deg' if spin else 'mm'}, "
                f"then made no progress for {self.stall_s:g} s (blocked?)"
            ))
            return 0.0, 0.0, 0.0
        accel = self.angular_accel if spin else self.accel
        least = min(abs(goal.rate), max(0.1 * abs(goal.rate), MIN_RAD_S if spin else MIN_MPS))
        speed = math.copysign(max(least, min(abs(goal.rate), math.sqrt(2 * accel * remaining))), goal.rate)
        if spin:
            return 0.0, 0.0, speed
        return 0.0, speed, HEADING_GAIN * (yaw0 - yaw)


def stage_modules(
    root: Any, chassis_path: str, pairs: Sequence[tuple[str, str]], forward_axis: str
) -> list[SwerveModule]:
    """Each module's geometry at steer 0, from the joints under `root` (a USD
    prim): the steer joint's anchor and axis, and the drive axle's rolling
    direction, in the chassis link's frame turned into the Viam frame. Only the
    joints' authored frames count (fixed joints up to the chassis link, matched
    by name), never the links' xforms, which hold whatever pose physics last
    wrote, so neither the base's motion nor the steer angles matter."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    from .stage_geometry import _pose

    stage = root.GetStage()
    meters = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    joints, by_child = {}, {}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint):
            joints[prim.GetName()] = prim
            targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
            if targets:
                by_child[str(targets[0])] = prim
    chassis = chassis_path.rsplit("/", 1)[-1]
    axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

    def joint(name: str) -> Any:
        prim = joints.get(name)
        if prim is None or not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f"swerve_modules: no revolute joint {name!r} under {root.GetPath()}")
        return UsdPhysics.RevoluteJoint(prim)

    def frames(j: Any) -> tuple[np.ndarray, np.ndarray]:
        """(joint frame in body0, joint frame in body1)."""
        return (
            _pose(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
            _pose(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()),
        )

    def body0_in_chassis(j: Any, name: str) -> np.ndarray:
        """The joint's parent link in the chassis frame, through the joints above it."""
        path, chain = str(j.GetBody0Rel().GetTargets()[0]), np.eye(4)
        while path.rsplit("/", 1)[-1] != chassis:
            above = by_child.get(path)
            if above is None:
                raise ValueError(f"swerve_modules: {name!r} doesn't hang off chassis_link {chassis!r}")
            f0, f1 = frames(UsdPhysics.Joint(above))
            chain = f0 @ np.linalg.inv(f1) @ chain
            path = str(UsdPhysics.Joint(above).GetBody0Rel().GetTargets()[0])
        return chain

    modules = []
    for steer_name, drive_name in pairs:
        steer, drive = joint(steer_name), joint(drive_name)
        s0, s1 = frames(steer)
        steer_frame = body0_in_chassis(steer, steer_name) @ s0  # the steer joint in the chassis frame
        steer_axis = steer_frame[:3, :3] @ axes[steer.GetAxisAttr().Get()]
        # the steer body at steer 0, then the drive joint's frame on it
        d0, _ = frames(drive)
        axle = (steer_frame @ np.linalg.inv(s1) @ d0)[:3, :3] @ axes[drive.GetAxisAttr().Get()]
        roll = np.cross(axle, (0.0, 0.0, 1.0))  # a positive drive speed's rolling direction
        rx, ry = to_viam_xy(roll[:2], forward_axis)
        px, py = to_viam_xy(steer_frame[:2, 3] * meters, forward_axis)
        modules.append(SwerveModule(
            steer=steer_name,
            drive=drive_name,
            position=(px, py),
            roll0=math.atan2(ry, rx),
            steer_sign=1.0 if steer_axis[2] > 0 else -1.0,
        ))
    return modules
