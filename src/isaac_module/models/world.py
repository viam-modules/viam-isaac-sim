"""erh:isaac-sim:world - the module's link to a running Isaac Sim.

Isaac Sim runs as its own process with the viam_isaac_server extension
enabled (see the README); this component connects to it. Configure exactly
one of these per machine. All other isaac-sim components name it in their
"world" attribute; their validate_config returns it as an implicit
dependency so viam-server connects the world first.

Attributes:
  mock (bool, default false)        - run without isaac sim (for dev/testing)
  address (string)                  - host:port of the viam_isaac_server
                                      extension, default localhost:47800
  connect_timeout_sec (float)       - how long to wait for the extension to
                                      answer, default 10
  props (list)                      - objects spawned into the scene unless a
                                      prim with that name already exists:
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

import asyncio
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
from ..protocol import parse_address
from ..sim_manager import DEFAULT_ADDRESS, SimConfig, SimManager

# attributes from when the module launched isaac sim itself; they now belong
# to whoever launches it
_LAUNCH_ATTRIBUTES = (
    "headless",
    "livestream",
    "livestream_public_ip",
    "livestream_width",
    "livestream_height",
    "usd_stage",
    "physics_dt",
    "rendering_dt",
    "boot_timeout_sec",
    "kit_log_level",
    "isaac_default_environment",
)


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
        parse_address(str(attrs.get("address", DEFAULT_ADDRESS)))
        if "connect_timeout_sec" in attrs and float(attrs["connect_timeout_sec"]) <= 0:
            raise ValueError("connect_timeout_sec must be positive")
        if not isinstance(attrs.get("props", []), list):
            raise ValueError("props must be a list")
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        ignored = [key for key in _LAUNCH_ATTRIBUTES if key in attrs]
        if ignored:
            self.logger.warning(
                "ignoring %s: isaac sim runs in its own process now, so set these "
                "when launching it",
                ", ".join(ignored),
            )
        cfg = SimConfig(
            mock=bool(attrs.get("mock", False)),
            address=str(attrs.get("address", DEFAULT_ADDRESS)),
            connect_timeout=float(attrs.get("connect_timeout_sec", 10.0)),
            props=[dict(p) for p in attrs.get("props", [])],
        )
        SimManager.get().configure(cfg)

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
            return await asyncio.to_thread(sim.status)
        if cmd == "play":
            await asyncio.to_thread(sim.play)
            return {"ok": True}
        if cmd == "pause":
            await asyncio.to_thread(sim.pause)
            return {"ok": True}
        if cmd == "reset":
            await asyncio.to_thread(sim.reset)
            return {"ok": True}
        if cmd == "add_usd":
            usd_path = str(command.get("usd_path", ""))
            prim_path = str(command.get("prim_path", ""))
            if not usd_path or not prim_path:
                raise ValueError("add_usd requires usd_path and prim_path")
            position = command.get("position") or [0.0, 0.0, 0.0]
            await asyncio.to_thread(
                sim.add_usd_reference,
                usd_path,
                prim_path,
                tuple(float(v) for v in position),
            )
            return {"ok": True}
        raise ValueError(
            f"unknown command {cmd!r}; supported: status, play, pause, reset, add_usd"
        )
