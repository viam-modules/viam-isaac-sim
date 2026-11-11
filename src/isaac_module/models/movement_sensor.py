"""viam-labs:isaac-sim:movement_sensor - GPS, IMU and compass readings from
a simulated rigid body.

Attributes:
  world (string, required)     - name of the viam-labs:isaac-sim:world component
  prop (string)                - the world prop to read (a dynamic prop)
  prim_path (string)           - or any rigid-body prim already in the stage
  origin ({latitude, longitude, altitude_m})
                               - geographic location of the sim world origin;
                                 world +X is east and +Y is north, so
                                 GetPosition reports real lat/lon. Default 0,0.
  position_noise_m (float)     - gaussian noise on the GPS fix, default 0
  heading_noise_deg (float)    - gaussian noise on the compass, default 0
  velocity_noise_mps (float)   - gaussian noise on linear velocity, default 0
  include_gravity (bool)       - IMU-style specific force (reads +9.81 on z at
                                 rest), default true; false for pure motion
  velocity_frame (string)      - "world" (default) or "base" (Viam base axes,
                                 +Y forward and +X right, taking body +X as
                                 forward), as wheel odometry consumers such
                                 as nav-stack SLAM expect
  seed (int)                   - fix the noise stream for reproducible runs

Readings:
  GetPosition           lat/lon from the origin; altitude is the body's z
  GetLinearVelocity     world frame (x east, y north, z up), m/s; or the
                        base frame with velocity_frame "base"
  GetAngularVelocity    body frame (roll, pitch, yaw rates), deg/s
  GetLinearAcceleration body frame, m/s^2
  GetCompassHeading     bearing of the body's +X axis, degrees clockwise from north
  GetOrientation        the body's world orientation as a Viam orientation vector
  GetReadings           all of the above plus sim_position_m, yaw_deg, speed_mps
"""

import asyncio
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.movement_sensor import MovementSensor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import GeoPoint, Orientation, ResourceName, Vector3
from viam.proto.component.movementsensor import GetAccuracyResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading

from .. import FAMILY, NAMESPACE
from ..bodies import MovementSensorHandle, MovementSensorSpec
from ..sim_manager import SimManager
from .motor import validate_body_attached
from .utils import get_attrs


class IsaacMovementSensor(MovementSensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "movement_sensor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[MovementSensorHandle] = None
        self._attrs: Dict[str, Any] = {}

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        sensor = cls(config.name)
        sensor.reconfigure(config, dependencies)
        return sensor

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        deps = validate_body_attached(config)
        MovementSensorSpec.from_attrs(get_attrs(config))
        return deps

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = get_attrs(config)
        self._attrs = attrs
        self._handle = SimManager.get().create_movement_sensor(self.name, attrs)

    def _h(self) -> MovementSensorHandle:
        if self._handle is None:
            raise RuntimeError(f"movement sensor {self.name} is not attached to the sim")
        return self._handle

    async def _readings(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._h().readings)

    async def get_position(self, **kwargs) -> Tuple[GeoPoint, float]:
        readings = await self._readings()
        position = readings["position"]
        return (
            GeoPoint(latitude=position["latitude"], longitude=position["longitude"]),
            float(position["altitude_m"]),
        )

    async def get_linear_velocity(self, **kwargs) -> Vector3:
        return Vector3(**(await self._readings())["linear_velocity"])

    async def get_angular_velocity(self, **kwargs) -> Vector3:
        return Vector3(**(await self._readings())["angular_velocity"])

    async def get_linear_acceleration(self, **kwargs) -> Vector3:
        return Vector3(**(await self._readings())["linear_acceleration"])

    async def get_compass_heading(self, **kwargs) -> float:
        return float((await self._readings())["compass_heading"])

    async def get_orientation(self, **kwargs) -> Orientation:
        return Orientation(**(await self._readings())["orientation"])

    async def get_properties(self, **kwargs) -> MovementSensor.Properties:
        return MovementSensor.Properties(
            linear_acceleration_supported=True,
            angular_velocity_supported=True,
            orientation_supported=True,
            position_supported=True,
            compass_heading_supported=True,
            linear_velocity_supported=True,
        )

    async def get_accuracy(self, **kwargs) -> GetAccuracyResponse:
        spec = self._h().spec
        return GetAccuracyResponse(
            accuracy={
                "position_m": spec.position_noise_m,
                "heading_deg": spec.heading_noise_deg,
                "velocity_mps": spec.velocity_noise_mps,
            },
            position_hdop=1.0 if spec.position_noise_m > 0.0 else 0.0,
            position_vdop=1.0 if spec.position_noise_m > 0.0 else 0.0,
            position_nmea_gga_fix=4,  # RTK fixed: the sim knows exactly where it is
            compass_degrees_error=spec.heading_noise_deg,
        )

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        return await self._readings()

    async def get_geometries(self, **kwargs):
        return []

    async def close(self) -> None:
        self._handle = None
        SimManager.get().remove_handle(self.name)
