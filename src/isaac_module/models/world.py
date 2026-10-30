"""viam-labs:isaac-sim:world - the generic component that owns the simulator.

Configure exactly one of these per machine. All other isaac-sim components
name it in their "world" attribute; their validate_config returns it as an
implicit dependency so viam-server boots the world first.

Attributes:
  mock (bool, default false)        - run without isaac sim (for dev/testing)
  headless (bool, default true)     - run kit without a local GUI window
  livestream (bool, default true)   - enable WebRTC livestreaming (view with
                                      the Isaac Sim WebRTC Streaming Client)
  dome_light (bool, default false)  - add a neutral dome light to the stage
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
  physics_dt (float)                - physics step size, default 1/60
  rendering_dt (float)              - render period; must be an integer
                                      multiple of physics_dt. Defaults to one
                                      render per capture sample
                                      (1/capture_rate_hz) when that is a whole
                                      number of physics steps, else physics_dt
  capture_rate_hz (float)           - episode capture rate, default 15
  capture_after_render (bool)       - sample episodes after each rendered
                                      frame with synchronous rendering, so
                                      images match joint state (default false)
  boot_timeout_sec (float)          - how long to wait for kit to boot
  kit_log_level (string)            - kit console verbosity, default "warning"
  props (list)                      - objects spawned into the scene at boot:
                                      {"name", "type": "cube"|"usd",
                                       "position": [x,y,z] meters,
                                       "size" (m), "scale" [sx,sy,sz],
                                       "color" [r,g,b] 0-1, "fixed" (bool),
                                       "visual_material": "wood"|"brushed_metal",
                                       "usd_path" (for type usd)}

DoCommand:
  {"command": "status"} | {"command": "play"} | {"command": "pause"} |
  {"command": "reset"} |
  {"command": "park_props", "props": ["pick_block"]} |
  {"command": "get_prop_pose", "prop": "pick_block"} |
  {"command": "set_prop_pose", "prop": "pick_block", "position": [x, y, z],
   "orientation_wxyz": [w, x, y, z]} |
  {"command": "start_episode", "output_dir": "runs/demo/episodes",
   "cosmos_output_dir": "runs/demo/cosmos", "episode_id": "000042",
   "seed": 42, "task": "Pick up the block", "arm": "pick-arm",
   "fixed_camera": "scene-cam", "fixed_camera_2": "third-cam",
   "wrist_camera": "wrist-cam",
   "target": {"prop_name": "pick_block", "position_m": [x,y,z],
              "rotation_wxyz": [w,x,y,z]},
   "nudge": {"amplitude_deg": 5, "interval_s": 1.2, "duration_s": 0.6}} |
  {"command": "mark_episode_stage", "task": "Lower the gripper"} |
  {"command": "stop_episode", "outcome": "success",
   "upload": {"dataset_id": "...", "tags": ["..."], "app_url": "..."}} |
  {"command": "upload_episode", "path": "runs/demo/episodes/000042",
   "upload": {"dataset_id": "...", "tags": ["..."],
              "cosmos_path": "runs/demo/cosmoswriter/000042"}} |
  {"command": "add_usd", "usd_path": "...", "prim_path": "/World/thing",
   "position": [x, y, z]}
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

from viam.components.generic import Generic
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .. import FAMILY, NAMESPACE
from ..dataset_upload import UploadRequest, upload_episode
from ..episode_schema import MINIMUM_STAGE_FRAMES
from ..nudge import NudgeConfig
from ..sim_manager import SimConfig, SimManager, validate_rendering_dt


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
    ) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        for key in (
            "physics_dt",
            "rendering_dt",
            "capture_rate_hz",
            "boot_timeout_sec",
            "livestream_width",
            "livestream_height",
        ):
            if key in attrs:
                value = float(attrs[key])
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{key} must be finite and positive")
        if "rendering_dt" in attrs:
            validate_rendering_dt(
                float(attrs.get("physics_dt", 1.0 / 60.0)),
                float(attrs["rendering_dt"]),
            )
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        cfg = SimConfig(
            mock=bool(attrs.get("mock", False)),
            headless=bool(attrs.get("headless", True)),
            livestream=bool(attrs.get("livestream", True)),
            dome_light=bool(attrs.get("dome_light", False)),
            usd_stage=attrs.get("usd_stage") or None,
            physics_dt=float(attrs.get("physics_dt", 1.0 / 60.0)),
            rendering_dt=(
                float(attrs["rendering_dt"])
                if attrs.get("rendering_dt") is not None
                else None
            ),
            capture_rate_hz=float(attrs.get("capture_rate_hz", 15.0)),
            capture_after_render=bool(attrs.get("capture_after_render", False)),
            boot_timeout=float(attrs.get("boot_timeout_sec", 300.0)),
            kit_log_level=str(attrs.get("kit_log_level", "warning")),
            livestream_public_ip=str(attrs.get("livestream_public_ip", "")),
            livestream_width=int(attrs.get("livestream_width", 1280)),
            livestream_height=int(attrs.get("livestream_height", 720)),
            props=[dict(p) for p in attrs.get("props", [])],
            isaac_default_environment=bool(attrs.get("isaac_default_environment", True)),
        )
        SimManager.get().ensure_booted(cfg)

    @staticmethod
    async def _wait_for_trainable_stage(sim: SimManager) -> None:
        """Let capture advance without blocking the serialized simulator thread."""
        while True:
            frame_count = await asyncio.to_thread(sim.active_episode_stage_frame_count)
            if frame_count is None or frame_count >= MINIMUM_STAGE_FRAMES:
                return
            capture_rate_hz = sim.cfg.capture_rate_hz if sim.cfg is not None else 15.0
            await asyncio.sleep(1.0 / capture_rate_hz)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
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
        if cmd == "park_props":
            props = command.get("props") or []
            return sim.park_props([str(name) for name in props])
        if cmd == "get_prop_pose":
            prop = str(command.get("prop", ""))
            if not prop:
                raise ValueError("get_prop_pose requires prop")
            return {"prop": prop, "pose": sim.get_prop_pose(prop)}
        if cmd == "set_prop_pose":
            prop = str(command.get("prop", ""))
            if not prop:
                raise ValueError("set_prop_pose requires prop")
            position = command.get("position")
            if not position:
                raise ValueError("set_prop_pose requires position")
            orientation = command.get("orientation_wxyz")
            if not orientation:
                raise ValueError("set_prop_pose requires orientation_wxyz")
            return sim.set_prop_pose(prop, position, orientation)
        if cmd == "start_episode":
            target = command.get("target")
            if not isinstance(target, Mapping):
                raise ValueError("start_episode requires target metadata")
            required = (
                "episode_id",
                "arm",
                "task",
                "gripper",
                "fixed_camera",
                "wrist_camera",
            )
            missing = [key for key in required if not command.get(key)]
            if missing:
                raise ValueError(f"start_episode missing required fields: {missing}")
            if not command.get("output_dir") and not command.get("cosmos_output_dir"):
                raise ValueError(
                    "start_episode requires output_dir or cosmos_output_dir"
                )
            nudge = command.get("nudge")
            if nudge is not None and not isinstance(nudge, Mapping):
                raise ValueError("start_episode nudge must be an object")
            return await asyncio.to_thread(
                sim.start_episode,
                str(command["output_dir"]) if command.get("output_dir") else None,
                str(command["episode_id"]),
                int(command.get("seed", 0)),
                str(command["task"]),
                str(command["arm"]),
                str(command["gripper"]),
                str(command["fixed_camera"]),
                str(command["wrist_camera"]),
                target,
                str(command["cosmos_output_dir"])
                if command.get("cosmos_output_dir")
                else None,
                str(command.get("fixed_camera_2", "")),
                NudgeConfig(**nudge) if nudge else None,
            )
        if cmd == "mark_episode_stage":
            task = str(command.get("task", "")).strip()
            if not task:
                raise ValueError("mark_episode_stage requires task")
            if not await asyncio.to_thread(sim.is_playing):
                raise RuntimeError(
                    "mark_episode_stage requires the simulation to be playing"
                )
            await self._wait_for_trainable_stage(sim)
            return await asyncio.to_thread(sim.mark_episode_stage, task)
        if cmd == "stop_episode":
            raw_upload = command.get("upload")
            await self._wait_for_trainable_stage(sim)
            result = await asyncio.to_thread(
                sim.stop_episode,
                str(command.get("outcome", "incomplete")),
            )
            if raw_upload:
                if not result.get("path"):
                    raise RuntimeError("episode upload requires native episode capture")
                upload_payload = dict(raw_upload)
                cosmos_dir = result.get("cosmos_output_dir")
                if cosmos_dir and not upload_payload.get("cosmos_path"):
                    upload_payload["cosmos_path"] = cosmos_dir
                upload = UploadRequest.from_command(upload_payload)
                result["upload"] = await upload_episode(
                    Path(result["path"]), upload, self.name
                )
            return result
        if cmd == "upload_episode":
            if not command.get("path"):
                raise ValueError("upload_episode requires path")
            return await upload_episode(
                Path(str(command["path"])),
                UploadRequest.from_command(command.get("upload")),
                self.name,
            )
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
            f"unknown command {cmd!r}; supported: status, play, pause, reset, "
            "park_props, get_prop_pose, set_prop_pose, start_episode, stop_episode, "
            "upload_episode, add_usd"
        )
