"""viam-labs:isaac-sim:base - a simulated differential-drive base.

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
from ..sim_manager import BaseHandle, SimManager
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component


class IsaacBase(Base, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "base")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[BaseHandle] = None
        self._attrs: Dict[str, Any] = {}
        self._max_linear = 0.5
        self._max_angular = 2.0

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
        return validate_sim_component(config)

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
        if velocity == 0:
            raise ValueError("velocity must be nonzero")
        meters = distance / 1000.0
        speed = abs(velocity) / 1000.0
        direction = 1.0 if (meters >= 0) == (velocity >= 0) else -1.0
        duration = abs(meters) / speed
        handle = self._h()
        await asyncio.to_thread(handle.set_velocity, direction * speed, 0.0)
        try:
            await asyncio.sleep(duration)
        finally:
            await asyncio.to_thread(handle.stop)

    async def spin(self, angle: float, velocity: float, **kwargs) -> None:
        if velocity == 0:
            raise ValueError("velocity must be nonzero")
        radians = math.radians(angle)
        speed = math.radians(abs(velocity))
        direction = 1.0 if (radians >= 0) == (velocity >= 0) else -1.0
        duration = abs(radians) / speed
        handle = self._h()
        await asyncio.to_thread(handle.set_velocity, 0.0, direction * speed)
        try:
            await asyncio.sleep(duration)
        finally:
            await asyncio.to_thread(handle.stop)

    async def set_power(self, linear: Vector3, angular: Vector3, **kwargs) -> None:
        lin = max(-1.0, min(1.0, linear.y)) * self._max_linear
        ang = max(-1.0, min(1.0, angular.z)) * self._max_angular
        await asyncio.to_thread(self._h().set_velocity, lin, ang)

    async def set_velocity(self, linear: Vector3, angular: Vector3, **kwargs) -> None:
        # viam: linear mm/s, angular deg/s -> isaac: m/s, rad/s
        await asyncio.to_thread(
            self._h().set_velocity, linear.y / 1000.0, math.radians(angular.z)
        )

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

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
