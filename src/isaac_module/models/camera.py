"""viam-labs:isaac-sim:camera - a simulated RGB camera.

Attributes:
  world (string, required)        - name of the viam-labs:isaac-sim:world component
  prim_path (string)              - existing camera prim to attach to, or
                                    where to create one (default /World/<name>)
  width / height (int)            - resolution, default 640x480
  position ([x,y,z] meters)       - where to place a newly created camera
  target ([x,y,z] meters)         - aim the camera at this point (easiest way
                                    to make a scene-monitor camera)
  orientation_rpy_deg ([r,p,y])   - explicit orientation instead of target
  fov_deg (float)                 - horizontal field of view; newly created
                                    cameras default to 70 (the usd default
                                    lens is a ~24 degree telephoto)
  parent_prim (string)            - create the camera as a child of this prim
                                    so it moves with it, e.g. an arm's wrist
                                    link for an end-effector camera
  local_position ([x,y,z] m)      - offset from parent_prim, default [0,0,0.05]
  local_orientation_rpy_deg       - orientation relative to parent_prim;
                                    default [180,0,0] = look out the +Z
                                    (tool) axis
"""

import asyncio
from io import BytesIO
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

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
from ..sim_manager import CameraHandle, SimManager
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component


class IsaacCamera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "camera")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[CameraHandle] = None
        self._attrs: Dict[str, Any] = {}

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        cam = cls(config.name)
        cam.reconfigure(config, dependencies)
        return cam

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        # cameras can always be created fresh, no asset/usd required
        return validate_sim_component(config, needs_source=False)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._attrs = attrs
        self._handle = SimManager.get().create_camera(self.name, attrs)

    def _h(self) -> CameraHandle:
        if self._handle is None:
            raise RuntimeError(f"camera {self.name} is not attached to the sim")
        return self._handle

    def _encode(self, mime_type: str = "") -> ViamImage:
        from PIL import Image

        arr = self._h().get_rgb()
        img = Image.fromarray(arr, mode="RGB")
        buf = BytesIO()
        if mime_type == CameraMimeType.PNG:
            img.save(buf, format="PNG")
            return ViamImage(buf.getvalue(), CameraMimeType.PNG)
        img.save(buf, format="JPEG", quality=90)
        return ViamImage(buf.getvalue(), CameraMimeType.JPEG)

    async def get_image(self, mime_type: str = "", **kwargs) -> ViamImage:
        return await asyncio.to_thread(self._encode, mime_type)

    async def get_images(
        self, **kwargs
    ) -> Tuple[List[NamedImage], ResponseMetadata]:
        image = await asyncio.to_thread(self._encode, CameraMimeType.JPEG)
        named = NamedImage(name=self.name, data=image.data, mime_type=image.mime_type)
        return [named], ResponseMetadata()

    async def get_point_cloud(self, **kwargs) -> Tuple[bytes, str]:
        raise NotImplementedError("point clouds are not supported yet")

    async def get_geometries(self, **kwargs):
        # cameras occupy no space; needed so the motion service can build a
        # world state when this camera has a frame
        return []

    async def get_properties(self, **kwargs) -> GetPropertiesResponse:
        return GetPropertiesResponse(supports_pcd=False)
