"""viam-labs:isaac-sim:base - a simulated differential-drive or holonomic base.

Attributes:
  world (string, required)     - name of the viam-labs:isaac-sim:world component
  asset (string)               - known robot, e.g. "jetbot" (brings sensible
                                 wheel defaults)
  usd_path (string)            - explicit robot USD to spawn
  prim_path (string)           - where to place it / existing robot prim
  position ([x,y,z] meters)    - spawn position
  wheel_joints ([left, right]) - wheel joint names (required unless the asset
                                 provides them)
  wheel_radius (meters)        - default from asset, else 0.05
  wheel_base (meters)          - default from asset, else 0.3
  max_linear_mps (float)       - full-power linear speed, default 0.5
  max_angular_rps (float)      - full-power angular speed (rad/s), default 2.0
  markers (list)               - visual-only colored squares riding on the
                                 chassis for cameras to find, e.g. fiducials
                                 for placing objects on the base:
                                 [{name, color: [r,g,b] 0-1, position:
                                 [x,y,z] m in the chassis link, size_m
                                 (0.03), thickness_m (0.002)}]
  chassis_link (string)        - link the markers ride on, under prim_path
                                 (the asset's, e.g. "chassis" for jetbot)
  planar_joints ([x, y, yaw])  - holonomic mode: drive these joints of the
                                 articulation an arm on prim_path loads (e.g.
                                 RidgebackFranka's dummy_base_* joints) instead of
                                 wheels; linear.x strafes. Add that arm to depends_on.
  swerve_modules (list)        - swerve mode: [{steer_joint, drive_joint}, ...] of
                                 the articulation an arm on prim_path loads (add it
                                 to depends_on); needs wheel_radius_m, optional
                                 forward_axis (+x/-x/+y/-y, default +x),
                                 chassis_link (default base_link) and
                                 wheel_drive_damping (N m s/rad, default 10).
                                 linear.x strafes; MoveStraight/Spin are closed loop.
"""

import asyncio
import math
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.base import Base
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..sim_manager import BaseHandle, SimManager, parse_markers
from ..swerve import parse_swerve_attrs
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component


class IsaacBase(Base, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "base")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[BaseHandle] = None
        self._attrs: Dict[str, Any] = {}
        self._max_linear = 0.5
        self._max_angular = 2.0
        self._commands = 0  # counts motion commands, so a timed hold sees a newer one

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        base = cls(config.name)
        base.reconfigure(config, dependencies)
        return base

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        deps = validate_sim_component(config)
        attrs = get_attrs(config)
        parse_markers(attrs.get("markers"))
        if "swerve_modules" in attrs:
            parse_swerve_attrs(attrs)
        return deps

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._max_linear = float(attrs.get("max_linear_mps", 0.5))
        self._max_angular = float(attrs.get("max_angular_rps", 2.0))
        self._attrs = attrs
        self._handle = SimManager.get().create_base(self.name, attrs)

    def _h(self) -> BaseHandle:
        if self._handle is None:
            raise RuntimeError(f"base {self.name} is not attached to the sim")
        return self._handle

    async def move_straight(self, distance: int, velocity: float, **kwargs) -> None:
        self._commands += 1
        if velocity == 0:
            raise ValueError("velocity must be nonzero")
        meters = distance / 1000.0
        speed = abs(velocity) / 1000.0
        direction = 1.0 if (meters >= 0) == (velocity >= 0) else -1.0
        if hasattr(self._h(), "run_goal"):
            return await self._goal("straight", direction * abs(meters), speed)
        duration = abs(meters) / speed
        await self._drive(direction * speed, 0.0, duration)

    async def spin(self, angle: float, velocity: float, **kwargs) -> None:
        self._commands += 1
        if velocity == 0:
            raise ValueError("velocity must be nonzero")
        radians = math.radians(angle)
        speed = math.radians(abs(velocity))
        direction = 1.0 if (radians >= 0) == (velocity >= 0) else -1.0
        if hasattr(self._h(), "run_goal"):
            return await self._goal("spin", direction * abs(radians), speed)
        duration = abs(radians) / speed
        await self._drive(0.0, direction * speed, duration)

    async def _goal(self, kind: str, target: float, speed: float) -> None:
        """Closed loop on the chassis pose (swerve): lands on target or raises."""
        handle = self._h()
        try:
            await asyncio.to_thread(handle.run_goal, kind, target, speed)
        finally:
            await asyncio.to_thread(handle.stop)

    async def _drive(self, linear: float, angular: float, duration: float) -> None:
        handle = self._h()
        try:
            if hasattr(handle, "drive_for"):  # bounded in sim time: exact distance
                await asyncio.to_thread(handle.drive_for, linear, angular, duration)
                while await asyncio.to_thread(handle.is_moving):
                    await asyncio.sleep(0.02)
            else:  # ponytail: wall-clock timing drifts when the sim runs off real time
                await asyncio.to_thread(handle.set_velocity, linear, angular)
                await asyncio.sleep(duration)
        finally:
            await asyncio.to_thread(handle.stop)

    async def set_power(self, linear: Vector3, angular: Vector3, **kwargs) -> None:
        self._commands += 1
        lin = max(-1.0, min(1.0, linear.y)) * self._max_linear
        ang = max(-1.0, min(1.0, angular.z)) * self._max_angular
        left = -max(-1.0, min(1.0, linear.x)) * self._max_linear  # viam +x is right
        await asyncio.to_thread(self._h().set_velocity, lin, ang, left)

    async def set_velocity(self, linear: Vector3, angular: Vector3, **kwargs) -> None:
        self._commands += 1
        # viam: linear mm/s, angular deg/s -> isaac: m/s, rad/s
        await asyncio.to_thread(
            self._h().set_velocity,
            linear.y / 1000.0,
            math.radians(angular.z),
            -linear.x / 1000.0,  # viam +x is right
        )

    async def stop(self, **kwargs) -> None:
        self._commands += 1
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def do_command(self, command: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """status (swerve): each module's steer and wheel state.
        set_velocity: SetVelocity's linear/angular ({x, y} mm/s, {z} deg/s) held
        for duration_s of sim time, then Stop; for clients whose SetVelocity is
        stopped as soon as the call (its session) ends."""
        handle = self._h()
        cmd = command.get("command")
        if cmd == "status" and hasattr(handle, "status"):
            return await asyncio.to_thread(handle.status)
        if cmd == "set_velocity":
            try:
                duration = float(command["duration_s"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("set_velocity needs duration_s (sim seconds)") from None
            if not (math.isfinite(duration) and duration > 0):
                raise ValueError("set_velocity duration_s must be a positive number")
            linear = Vector3(**{k: float(v) for k, v in (command.get("linear") or {}).items()})
            angular = Vector3(**{k: float(v) for k, v in (command.get("angular") or {}).items()})
            sim = SimManager.get()
            await self.set_velocity(linear, angular)
            mine = self._commands
            try:
                start = sim.sim_time_s()
                # until the time is up, another command replaces this one, or the base stops
                while self._commands == mine and sim.sim_time_s() - start < duration:
                    if not await self.is_moving():
                        break
                    await asyncio.sleep(0.02)
            finally:
                if self._commands == mine:
                    await self.stop()
            return {}
        raise ValueError(f"unknown command {cmd!r}; supported: status (swerve), set_velocity")

    async def get_geometries(self, **kwargs):
        # needed so the motion service can build a world state when this
        # base has a frame; not modeling the chassis yet
        return []

    async def get_properties(self, **kwargs) -> Base.Properties:
        wheel_radius = getattr(self._h(), "wheel_radius", 0.05)
        wheel_base = getattr(self._h(), "wheel_base", 0.3)
        return Base.Properties(
            width_meters=wheel_base,
            turning_radius_meters=0.0,
            wheel_circumference_meters=2 * math.pi * wheel_radius,
        )
