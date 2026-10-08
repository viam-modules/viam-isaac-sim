"""viam-labs:isaac-sim:world - the generic component that owns the simulator.

Configure exactly one of these per machine. All other isaac-sim components
name it in their "world" attribute; their validate_config returns it as an
implicit dependency so viam-server boots the world first.

Attributes:
  mock (bool, default false)        - run without isaac sim (for dev/testing)
  headless (bool, default true)     - run kit without a local GUI window
  livestream (bool, default true)   - enable WebRTC livestreaming (view with
                                      the Isaac Sim WebRTC Streaming Client)
  livestream_public_ip (string)     - IP advertised to streaming clients;
                                      auto-detected if unset
  livestream_width / _height (int)  - streamed resolution, default 1280x720
  usd_stage (string)                - USD file/omniverse URL to open; if unset
                                      an empty stage with a ground plane is used
  isaac_default_environment (bool)  - default true. When false, skip Isaac's
                                      default environment (which bundles a
                                      floor collider AND the scene lighting)
                                      and instead add a plain DistantLight +
                                      DomeLight. Use this when the arm needs
                                      to reach negative z; the default floor
                                      otherwise pins it
  physics_dt / rendering_dt (float) - sim step sizes, default 1/60
  boot_timeout_sec (float)          - how long to wait for kit to boot
  kit_log_level (string)            - kit console verbosity, default "warning"
  props (list)                      - objects spawned into the scene at boot:
                                      {"name", "type": "cube"|"usd",
                                       "position": [x,y,z] meters,
                                       "size" (m), "scale" [sx,sy,sz],
                                       "color" [r,g,b] 0-1, "fixed" (bool),
                                       "usd_path" (for type usd)}

DoCommand:
  {"command": "status"} | {"command": "play"} | {"command": "pause"} |
  {"command": "reset"} |
  {"command": "add_usd", "usd_path": "...", "prim_path": "/World/thing",
   "position": [x, y, z]}
"""

from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.generic import Generic
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .. import FAMILY, NAMESPACE
from ..sim_manager import SimConfig, SimManager


class IsaacWorld(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "world")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        w = cls(config.name)
        w.reconfigure(config, dependencies)
        return w

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        for key in (
            "physics_dt",
            "rendering_dt",
            "boot_timeout_sec",
            "livestream_width",
            "livestream_height",
        ):
            if key in attrs and float(attrs[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        cfg = SimConfig(
            mock=bool(attrs.get("mock", False)),
            headless=bool(attrs.get("headless", True)),
            livestream=bool(attrs.get("livestream", True)),
            usd_stage=attrs.get("usd_stage") or None,
            physics_dt=float(attrs.get("physics_dt", 1.0 / 60.0)),
            rendering_dt=float(attrs.get("rendering_dt", 1.0 / 60.0)),
            boot_timeout=float(attrs.get("boot_timeout_sec", 300.0)),
            kit_log_level=str(attrs.get("kit_log_level", "warning")),
            livestream_public_ip=str(attrs.get("livestream_public_ip", "")),
            livestream_width=int(attrs.get("livestream_width", 1280)),
            livestream_height=int(attrs.get("livestream_height", 720)),
            props=[dict(p) for p in attrs.get("props", [])],
            isaac_default_environment=bool(attrs.get("isaac_default_environment", True)),
        )
        SimManager.get().ensure_booted(cfg)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        sim = SimManager.get()
        cmd = str(command.get("command", ""))
        if cmd == "status":
            return sim.status()
        if cmd == "play":
            sim.play()
            return {"ok": True}
        if cmd == "pause":
            sim.pause()
            return {"ok": True}
        if cmd == "reset":
            sim.reset()
            return {"ok": True}
        if cmd == "add_usd":
            usd_path = str(command.get("usd_path", ""))
            prim_path = str(command.get("prim_path", ""))
            if not usd_path or not prim_path:
                raise ValueError("add_usd requires usd_path and prim_path")
            position = command.get("position") or [0.0, 0.0, 0.0]
            sim.add_usd_reference(
                usd_path, prim_path, tuple(float(v) for v in position)
            )
            return {"ok": True}
        raise ValueError(
            f"unknown command {cmd!r}; supported: status, play, pause, reset, add_usd"
        )
