"""viam-labs:isaac-sim:camera - a simulated RGB camera.

Attributes:
  world (string, required)        - name of the viam-labs:isaac-sim:world component
  prim_path (string)              - existing camera prim to attach to, or
                                    where to create one (default /World/<name>)
  width / height (int)            - resolution, default 640x480
  position ([x,y,z] meters)       - legacy placement when no standard frame is set
  target ([x,y,z] meters)         - aim the camera at this point (easiest way
                                    to make a scene-monitor camera)
  orientation_rpy_deg ([r,p,y])   - legacy orientation when no frame is set
  fov_deg (float)                 - horizontal field of view; newly created
                                    cameras default to 70 (the usd default
                                    lens is a ~24 degree telephoto)
  near_clip_m (float)             - near clipping distance, default 0.01 m
  parent_prim (string)            - create the camera as a child of this prim
                                    so it moves with it, e.g. an arm's wrist
                                    link for an end-effector camera
  local_position / local_orientation_rpy_deg
                                  - legacy pose relative to parent_prim
  parent_prim_rpy_deg ([r,p,y])   - with parent_prim and a frame: orientation of
                                    the frame's parent within parent_prim, when
                                    they differ (e.g. an arm's end frame vs its
                                    USD flange link), so the camera lands where
                                    the frame system thinks it is
  depth (bool)                    - also serve depth (image/vnd.viam.dep) and
                                    point clouds; restart to change. get_images
                                    sources are then "color" and "depth"
                                    (else a single source named after the
                                    component)
  max_depth_m (float)             - drop depth beyond this many meters
  depth_first (bool)              - with depth: return "depth" before "color",
                                    for consumers that decode only the first
                                    image (e.g. obstacles-depth), like a
                                    RealSense with depth listed first
  standard frame                  - preferred placement; conflicting legacy
                                    pose attributes are rejected

Frame-derived pose, target, fov_deg, and near_clip_m changes apply live between
episodes. prim_path, parent_prim, width, and height changes require a module
restart.
"""

import asyncio
from io import BytesIO
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata, ResourceName
from viam.proto.component.camera import GetPropertiesResponse, IntrinsicParameters
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..pointcloud import encode_depth, optical_to_axes, to_pcd, unproject
from ..sim_manager import CameraHandle, SimManager, camera_axes
from .utils import (
    apply_camera_frame_to_attrs,
    camera_frame_orientation,
    get_attrs,
    validate_sim_component,
)


class IsaacCamera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "camera")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[CameraHandle] = None
        self._attrs: Dict[str, Any] = {}
        self._depth = False
        self._max_depth_m: Optional[float] = None
        self._depth_first = False

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
        attrs = get_attrs(config)
        if config.HasField("frame"):
            parented = bool(attrs.get("parent_prim")) and config.frame.parent != "world"
            position_key = "local_position" if parented else "position"
            if attrs.get(position_key) is not None:
                raise ValueError(
                    f"{config.name}: attributes.{position_key} conflicts with "
                    "frame.translation; use the standard frame"
                )
            orientation_keys = (
                ("local_orientation_wxyz", "local_orientation_rpy_deg")
                if parented
                else ("orientation_wxyz", "orientation_rpy_deg")
            )
            has_orientation = (
                config.frame.orientation.WhichOneof("type") is not None
                if parented
                else camera_frame_orientation(config) is not None
            )
            if has_orientation:
                conflicting = [key for key in orientation_keys if attrs.get(key) is not None]
                if conflicting:
                    raise ValueError(
                        f"{config.name}: attributes.{conflicting[0]} conflicts with "
                        "frame.orientation; use the standard frame"
                    )
                if attrs.get("target") is not None:
                    raise ValueError(
                        f"{config.name}: attributes.target conflicts with "
                        "frame.orientation; choose one orientation source"
                    )
        if attrs.get("parent_prim") and attrs.get("target") is not None:
            raise ValueError(
                f"{config.name}: attributes.target is unsupported with parent_prim"
            )
        max_depth = attrs.get("max_depth_m")
        if max_depth is not None:
            if not attrs.get("depth"):
                raise ValueError(f'{config.name}: max_depth_m needs "depth": true')
            if float(max_depth) <= 0:
                raise ValueError(f"{config.name}: max_depth_m must be > 0")
        if attrs.get("depth_first") and not attrs.get("depth"):
            raise ValueError(f'{config.name}: depth_first needs "depth": true')
        # cameras can always be created fresh, no asset/usd required
        return validate_sim_component(config, needs_source=False)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_camera_frame_to_attrs(config, get_attrs(config))
        # viam-side only: never sent to the sim, so it changes even mid-episode
        max_depth = attrs.pop("max_depth_m", None)
        depth_first = bool(attrs.pop("depth_first", False))
        # create_camera may raise (restart required); only commit state after it
        self._handle = SimManager.get().create_camera(self.name, attrs)
        self._max_depth_m = float(max_depth) if max_depth is not None else None
        self._depth = bool(attrs.get("depth", False))
        self._depth_first = depth_first
        self._attrs = attrs

    def _h(self) -> CameraHandle:
        if self._handle is None:
            raise RuntimeError(f"camera {self.name} is not attached to the sim")
        return self._handle

    @staticmethod
    def _encode_rgb(arr, mime_type: str = "") -> ViamImage:
        from PIL import Image

        img = Image.fromarray(arr, mode="RGB")
        buf = BytesIO()
        if mime_type == CameraMimeType.PNG:
            img.save(buf, format="PNG")
            return ViamImage(buf.getvalue(), CameraMimeType.PNG)
        img.save(buf, format="JPEG", quality=90)
        return ViamImage(buf.getvalue(), CameraMimeType.JPEG)

    async def get_image(self, mime_type: str = "", **kwargs) -> ViamImage:
        return await asyncio.to_thread(
            lambda: self._encode_rgb(self._h().get_rgb(), mime_type)
        )

    def _named_images(self, names: Optional[Sequence[str]] = None) -> List[NamedImage]:
        # depth on: "color"/"depth" (RealSense convention, what
        # detections-to-segments expects); off: the component name.
        color_name = "color" if self._depth else self.name
        want = set(names or [color_name, "depth"])
        want_color = color_name in want
        want_depth = self._depth and "depth" in want
        h = self._h()
        rgb, depth = h.get_rgbd() if want_depth else (h.get_rgb(), None)
        out = []
        if want_color:
            color = self._encode_rgb(rgb)
            out.append(NamedImage(color_name, color.data, color.mime_type))
        if want_depth:
            out.insert(
                0 if self._depth_first else len(out),
                NamedImage(
                    "depth",
                    encode_depth(depth, self._max_depth_m),
                    CameraMimeType.VIAM_RAW_DEPTH,
                ),
            )
        return out

    async def get_images(
        self, *, filter_source_names: Optional[Sequence[str]] = None, **kwargs
    ) -> Tuple[List[NamedImage], ResponseMetadata]:
        images = await asyncio.to_thread(self._named_images, filter_source_names)
        return images, ResponseMetadata()

    def _point_cloud(self) -> bytes:
        handle = self._h()
        rgb, depth = handle.get_rgbd()
        points, colors = unproject(depth, handle.intrinsics(), rgb, self._max_depth_m)
        return to_pcd(optical_to_axes(points, camera_axes(self._attrs)), colors)

    async def get_point_cloud(self, **kwargs) -> Tuple[bytes, str]:
        if not self._depth:
            raise NotImplementedError(f'{self.name}: point clouds need "depth": true')
        return await asyncio.to_thread(self._point_cloud), CameraMimeType.PCD

    async def get_geometries(self, **kwargs):
        # cameras occupy no space; needed so the motion service can build a
        # world state when this camera has a frame
        return []

    async def get_properties(self, **kwargs) -> GetPropertiesResponse:
        fx, fy, cx, cy = await asyncio.to_thread(self._h().intrinsics)
        return GetPropertiesResponse(
            supports_pcd=self._depth,
            intrinsic_parameters=IntrinsicParameters(
                width_px=int(self._attrs.get("width", 640)),
                height_px=int(self._attrs.get("height", 480)),
                focal_x_px=fx,
                focal_y_px=fy,
                center_x_px=cx,
                center_y_px=cy,
            ),
        )
