"""viam-labs:isaac-sim:lidar - a simulated RTX lidar served as a point-cloud
camera (Viam has no lidar API; this matches e.g. the RPLidar module).

Attributes:
  world (string, required)  - name of the viam-labs:isaac-sim:world component
  sensor (string, required) - preset: rplidar_s2e, ouster_os1_32,
                              robosense_vega (OS1-32 stand-in)
  parent_prim (string)      - mount on this prim so the lidar moves with it;
                              the standard frame is then relative to it
  min_range_m / max_range_m - drop returns outside this range (meters)
  rate_hz (float)           - publish rate, at most the preset's native rate
  standard frame            - placement; lidar axes are X forward, Z up

sensor and parent_prim changes need a module restart; frame, ranges and
rate_hz apply live. Points are meters in the lidar frame.
"""

import asyncio
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata, ResourceName
from viam.proto.component.camera import GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..lidar import LIDAR_PRESETS, LidarHandle, effective_rate
from ..pointcloud import filter_range, to_pcd
from ..sim_manager import SimManager
from .utils import apply_lidar_frame_to_attrs, get_attrs, validate_sim_component


class IsaacLidar(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "lidar")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[LidarHandle] = None
        self._min_range_m = 0.0
        self._max_range_m: Optional[float] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        lidar = cls(config.name)
        lidar.reconfigure(config, dependencies)
        return lidar

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = get_attrs(config)
        name = config.name
        sensor = attrs.get("sensor")
        if sensor not in LIDAR_PRESETS:
            raise ValueError(f'{name}: set "sensor" to one of {sorted(LIDAR_PRESETS)}')
        if attrs.get("rate_hz") is not None and float(attrs["rate_hz"]) <= 0:
            raise ValueError(f"{name}: rate_hz must be > 0")
        try:
            effective_rate(attrs)
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from None
        min_r = float(attrs.get("min_range_m", 0.0))
        max_r = attrs.get("max_range_m")
        if min_r < 0:
            raise ValueError(f"{name}: min_range_m must be >= 0")
        if max_r is not None:
            if float(max_r) <= 0:
                raise ValueError(f"{name}: max_range_m must be > 0")
            if min_r >= float(max_r):
                raise ValueError(f"{name}: min_range_m must be < max_range_m")
        if (
            attrs.get("parent_prim")
            and config.HasField("frame")
            and config.frame.parent == "world"
        ):
            raise ValueError(
                f'{name}: with parent_prim, set frame.parent to the component '
                'that prim belongs to (the frame is relative to it), not "world"'
            )
        return validate_sim_component(config, needs_source=False)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_lidar_frame_to_attrs(config, get_attrs(config))
        # viam-side filters: never sent to the sim
        min_r = float(attrs.pop("min_range_m", 0.0))
        max_r = attrs.pop("max_range_m", None)
        # create_lidar may raise (restart required); only commit state after it
        self._handle = SimManager.get().create_lidar(self.name, attrs)
        self._min_range_m = min_r
        self._max_range_m = float(max_r) if max_r is not None else None

    def _h(self) -> LidarHandle:
        if self._handle is None:
            raise RuntimeError(f"lidar {self.name} is not attached to the sim")
        return self._handle

    def _point_cloud(self) -> bytes:
        scan = self._h().latest_scan()
        if scan is None:
            raise RuntimeError(
                f"{self.name}: no lidar scan yet - is the simulation playing?"
            )
        return to_pcd(filter_range(scan[1], self._min_range_m, self._max_range_m))

    async def close(self) -> None:
        self._handle = None
        await asyncio.to_thread(SimManager.get().remove_lidar, self.name)

    async def get_point_cloud(self, **kwargs) -> Tuple[bytes, str]:
        return await asyncio.to_thread(self._point_cloud), CameraMimeType.PCD

    async def get_image(self, mime_type: str = "", **kwargs) -> ViamImage:
        raise NotImplementedError(f"{self.name}: lidar has no image")

    async def get_images(self, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        raise NotImplementedError(f"{self.name}: lidar has no image")

    async def get_geometries(self, **kwargs):
        return []

    async def get_properties(self, **kwargs) -> GetPropertiesResponse:
        return GetPropertiesResponse(supports_pcd=True)
