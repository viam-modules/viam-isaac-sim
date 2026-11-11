"""viam-labs:isaac-sim:scene_camera - an observer camera you can orbit.

Same attributes as viam-labs:isaac-sim:camera, except:
  - parent_prim, orientation_wxyz and orientation_rpy_deg are rejected;
    the camera always looks at `target`
  - a standard frame sets the home position; its orientation must be
    identity (the app editor's default)
  - position defaults to [3, 3, 2.5], target to [0, 0, 0], fov_deg to 70

Config position + target + fov_deg is the home view.

DoCommand contract (View = {target, distance, yaw_deg, pitch_deg, fov_deg,
home: {same five keys}, recording}):
  {"command": "get_view"}                 -> View
  {"command": "set_view", <any of target, distance, yaw_deg, pitch_deg,
   fov_deg>}                              -> View (merged, clamped)
  {"command": "reset_view"}               -> View (back to home)
  {"command": "get-custom-control"}       -> {html, revision, path}

Clamps: distance >= 0.05 m, pitch within +-89 deg, yaw wrapped to
[-180, 180], fov_deg within [1, 179]. Views cannot change while an episode
is recording (the existing camera capture guard raises). A flown view lasts
until the component's config changes or the module restarts; the frame
system keeps reporting the config pose.
"""

import asyncio
import math
import threading
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Sequence, Tuple

from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..custom_control import handle_custom_control
from ..sim_manager import SimManager, _quaternions_close
from ..spatial import Vec3, orbit_to_position, position_to_orbit, to_vec3
from .camera import IsaacCamera
from .utils import apply_camera_frame_to_attrs, frame_pose, get_attrs, validate_sim_component

# module root: src/isaac_module/models/scene_camera.py -> parents[3]
META_PATH = Path(__file__).resolve().parents[3] / "meta.json"

DEFAULT_POSITION: Vec3 = (3.0, 3.0, 2.5)
DEFAULT_TARGET: Vec3 = (0.0, 0.0, 0.0)
DEFAULT_FOV_DEG = 70.0
MIN_DISTANCE = 0.05
MAX_PITCH_DEG = 89.0
VIEW_KEYS = ("target", "distance", "yaw_deg", "pitch_deg", "fov_deg")


def _home_pose(config: ComponentConfig) -> Tuple[Vec3, Vec3]:
    """Home (position, target), shared by validate_config and reconfigure so
    both resolve defaults the same way: frame translation, else
    attributes.position, else DEFAULT_POSITION."""
    attrs = get_attrs(config)
    position, _ = frame_pose(config)
    if position is None:
        position = to_vec3(attrs.get("position"), DEFAULT_POSITION)
    return position, to_vec3(attrs.get("target"), DEFAULT_TARGET)


def _clamp_view(view: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target": [float(v) for v in view["target"]],
        "distance": max(MIN_DISTANCE, float(view["distance"])),
        "yaw_deg": math.remainder(float(view["yaw_deg"]), 360.0),
        "pitch_deg": max(-MAX_PITCH_DEG, min(MAX_PITCH_DEG, float(view["pitch_deg"]))),
        "fov_deg": max(1.0, min(179.0, float(view["fov_deg"]))),
    }


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _parse_patch(patch: Mapping[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(patch) - set(VIEW_KEYS))
    if unknown:
        raise ValueError(
            f"set_view: unknown keys {unknown}; supported: {', '.join(VIEW_KEYS)}"
        )
    parsed: Dict[str, Any] = {}
    for key, value in patch.items():
        if key == "target":
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or not all(_is_number(v) for v in value)
            ):
                raise ValueError("set_view: target must be [x, y, z]")
            parsed[key] = [float(v) for v in value]
            continue
        if not _is_number(value):
            raise ValueError(f"set_view: {key} must be a finite number")
        parsed[key] = float(value)
    return parsed


class IsaacSceneCamera(IsaacCamera):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "scene_camera")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        # guards merge-then-apply so two clients cannot interleave
        self._lock = threading.Lock()
        self._home: Dict[str, Any] = {}
        self._view: Dict[str, Any] = {}

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        name = config.name
        attrs = get_attrs(config)
        if attrs.get("parent_prim"):
            raise ValueError(
                f"{name}: scene_camera does not support parent_prim; use "
                "viam-labs:isaac-sim:camera for a mounted camera"
            )
        for key in ("orientation_wxyz", "orientation_rpy_deg"):
            if attrs.get(key) is not None:
                raise ValueError(
                    f"{name}: attributes.{key} is unsupported; scene_camera "
                    "aims with attributes.target"
                )
        if config.HasField("frame"):
            if attrs.get("position") is not None:
                raise ValueError(
                    f"{name}: attributes.position conflicts with "
                    "frame.translation; use the standard frame"
                )
            _, quat = frame_pose(config)
            if quat is not None and not _quaternions_close(quat, (1.0, 0.0, 0.0, 0.0)):
                raise ValueError(
                    f"{name}: frame.orientation must be identity; scene_camera "
                    "aims with attributes.target"
                )
        position, target = _home_pose(config)
        try:
            distance, _, pitch = position_to_orbit(position, target)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        # home must already satisfy _clamp_view, or reset_view would move the camera
        if distance < MIN_DISTANCE:
            raise ValueError(
                f"{name}: position must be at least {MIN_DISTANCE} m from target"
            )
        if abs(pitch) > MAX_PITCH_DEG:
            raise ValueError(
                f"{name}: the home view looks {abs(pitch):.1f} deg up or down; "
                f"keep it within {MAX_PITCH_DEG:.0f} deg so it can orbit"
            )
        fov = attrs.get("fov_deg", DEFAULT_FOV_DEG)
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not 1.0 <= float(fov) <= 179.0:
            raise ValueError(f"{name}: fov_deg must be between 1 and 179")
        return validate_sim_component(config, needs_source=False)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_camera_frame_to_attrs(config, get_attrs(config))
        attrs.pop("orientation_wxyz", None)
        position, target = _home_pose(config)
        # always present, so later view changes only touch live-mutable keys
        attrs["position"] = list(position)
        attrs["target"] = list(target)
        attrs["fov_deg"] = float(attrs.get("fov_deg", DEFAULT_FOV_DEG))
        distance, yaw, pitch = position_to_orbit(position, target)
        home = _clamp_view(
            {"target": target, "distance": distance, "yaw_deg": yaw,
             "pitch_deg": pitch, "fov_deg": attrs["fov_deg"]}
        )
        with self._lock:
            self._handle = SimManager.get().create_camera(self.name, attrs)
            self._attrs = attrs
            self._home = home
            self._view = dict(home)

    # view methods are synchronous (they block on the sim thread);
    # do_command runs them via asyncio.to_thread

    def _apply(self, view: Dict[str, Any]) -> None:
        position = orbit_to_position(
            view["target"], view["distance"], view["yaw_deg"], view["pitch_deg"]
        )
        attrs = {
            **self._attrs,
            "position": list(position),
            "target": list(view["target"]),
            "fov_deg": view["fov_deg"],
        }
        # if SimManager.run times out the queued update still runs later, so
        # _handles can briefly disagree with _view until the next set_view (accepted)
        self._handle = SimManager.get().create_camera(self.name, attrs)

    def get_view(self) -> Dict[str, Any]:
        # lock-free: _view/_home are only ever replaced whole, never mutated,
        # so a poll never waits behind a slow set_view
        view, home = self._view, self._home
        view = {**view, "target": list(view["target"])}
        home = {**home, "target": list(home["target"])}
        return {**view, "home": home, "recording": SimManager.get().episode_active}

    def set_view(self, patch: Mapping[str, Any]) -> Dict[str, Any]:
        changes = _parse_patch(patch)
        with self._lock:
            view = _clamp_view({**self._view, **changes})
            self._apply(view)  # raises during episode capture; view unchanged
            self._view = view
        return self.get_view()

    def reset_view(self) -> Dict[str, Any]:
        with self._lock:
            self._apply(self._home)
            self._view = dict(self._home)
        return self.get_view()

    async def do_command(
        self, command: Mapping[str, Any], *, timeout=None, **kwargs
    ) -> Mapping[str, Any]:
        served = handle_custom_control(command, self.MODEL, META_PATH)
        if served is not None:
            return served
        cmd = str(command.get("command", ""))
        if cmd == "get_view":
            return await asyncio.to_thread(self.get_view)
        if cmd == "set_view":
            patch = {k: v for k, v in command.items() if k != "command"}
            return await asyncio.to_thread(self.set_view, patch)
        if cmd == "reset_view":
            return await asyncio.to_thread(self.reset_view)
        raise ValueError(
            f"unknown command {cmd!r}; supported: get_view, set_view, "
            "reset_view, get-custom-control"
        )
