"""The Isaac Sim half of the server: attaches viam components to prims in
the running stage and drives them.

Every rpc_* method runs on Kit's main thread (see server.py). The host app
owns the stage and the timeline: nothing here opens a stage, steps physics
or resets the world on its own. Components attach to the prims they name and
only spawn what isn't there yet, so re-attaching (module restarts,
reconfigures) never duplicates anything.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom

from .images import encode_rgb
from .server import Binary, NotAttached, NotReady

LOGGER = logging.getLogger("viam_isaac_server")

# Assets shipped on the Isaac Sim content server, addressable by a short name
# in component config. Paths are relative to the assets root; where isaac 5.0
# moved an asset, the 5.0 path is listed first with the 4.5 path as a
# fallback - the first candidate that exists is used.
KNOWN_ASSETS: Dict[str, Dict[str, Any]] = {
    "ur3e": {"usd": ["/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"]},
    "ur5e": {"usd": ["/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"]},
    "ur10": {"usd": ["/Isaac/Robots/UniversalRobots/ur10/ur10.usd"]},
    "ur10e": {"usd": ["/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"]},
    "ur16e": {"usd": ["/Isaac/Robots/UniversalRobots/ur16e/ur16e.usd"]},
    "ur20": {"usd": ["/Isaac/Robots/UniversalRobots/ur20/ur20.usd"]},
    "franka": {
        "usd": [
            "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "/Isaac/Robots/Franka/franka.usd",
        ]
    },
    "jetbot": {
        "usd": [
            "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
            "/Isaac/Robots/Jetbot/jetbot.usd",
        ],
        "wheel_joints": ["left_wheel_joint", "right_wheel_joint"],
        "wheel_radius": 0.03,
        "wheel_base": 0.1125,
    },
}

_STOPPED = (
    "the simulation is stopped; press Play in Isaac Sim or send the world "
    '{"command": "play"}'
)


class _Articulated:
    """An articulation whose physics handle is (re)created on first use in
    each play session - Isaac invalidates physics handles whenever the
    timeline stops, and the host app can stop and play at any time."""

    def __init__(
        self, backend: "IsaacBackend", prim_path: str, articulation: Any
    ) -> None:
        self.prim_path = prim_path
        self.articulation = articulation
        self._backend = backend
        self._session: Optional[int] = None

    @property
    def initialized(self) -> bool:
        return self._session == self._backend.session

    def invalidate(self) -> None:
        self._session = None

    async def ready(self) -> Any:
        if self._backend.timeline.is_stopped():
            raise NotReady(_STOPPED)
        if not self.initialized:
            error = self._try_initialize()
            if error is not None:
                # physics creates handles for new prims (and its views after
                # Play) on the next update
                await _next_update()
                error = self._try_initialize()
            if error is not None:
                raise NotReady(
                    f"{self.prim_path}: articulation not ready ({error}); is it an "
                    "articulation root?"
                )
            self._session = self._backend.session
        return self.articulation

    def _try_initialize(self) -> Optional[Exception]:
        try:
            self.articulation.initialize()
            if self.articulation.get_joint_positions() is None:
                return RuntimeError("no joint state")
        except Exception as e:
            return e
        return None


@dataclass
class _Arm:
    prim_path: str
    stage_id: int
    body: _Articulated
    ee: Any


@dataclass
class _Camera:
    prim_path: str
    stage_id: int
    camera: Any
    resolution: Tuple[int, int]


@dataclass
class _Base:
    prim_path: str
    stage_id: int
    body: _Articulated
    controller: Any
    wheel_radius: float
    wheel_base: float
    cmd: Tuple[float, float] = (0.0, 0.0)
    # connection that last commanded it; its disconnect stops the base
    owner: Optional[int] = None


class IsaacBackend:
    def __init__(self) -> None:
        self.timeline = omni.timeline.get_timeline_interface()
        # bumped whenever the timeline stops, which invalidates every physics
        # handle; articulations re-initialize lazily in the next session
        self.session = 0
        self._arms: Dict[str, _Arm] = {}
        self._cameras: Dict[str, _Camera] = {}
        self._bases: Dict[str, _Base] = {}
        self._timeline_sub = (
            self.timeline.get_timeline_event_stream().create_subscription_to_pop(
                self._on_timeline_event, name="viam_isaac_server"
            )
        )
        self._physics_sub = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                self._on_physics_step
            )
        )

    def shutdown(self) -> None:
        self._timeline_sub = None
        self._physics_sub = None
        self._arms.clear()
        self._cameras.clear()
        self._bases.clear()

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def _on_timeline_event(self, event: Any) -> None:
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.session += 1
            for base in self._bases.values():
                base.cmd = (0.0, 0.0)

    def _on_physics_step(self, step_size: float) -> None:
        for name, base in list(self._bases.items()):
            if not base.body.initialized:
                continue
            try:
                base.body.articulation.apply_wheel_actions(
                    base.controller.forward(command=list(base.cmd))
                )
            except Exception:
                LOGGER.exception("error driving base %s; stopping it", name)
                base.cmd = (0.0, 0.0)
                base.body.invalidate()

    def on_disconnect(self, ctx: Any) -> None:
        for name, base in self._bases.items():
            if base.owner == ctx.id and base.cmd != (0.0, 0.0):
                LOGGER.warning("stopping base %s: its viam module disconnected", name)
                base.cmd = (0.0, 0.0)

    # ------------------------------------------------------------------
    # world
    # ------------------------------------------------------------------

    def rpc_world_status(self, ctx: Any) -> Dict[str, Any]:
        stage = _stage()
        return {
            "playing": bool(self.timeline.is_playing()),
            "stopped": bool(self.timeline.is_stopped()),
            "sim_time": float(self.timeline.get_current_time()),
            "stage": stage.GetRootLayer().identifier if stage else "",
            "meters_per_unit": (
                float(UsdGeom.GetStageMetersPerUnit(stage)) if stage else 1.0
            ),
            "arms": sorted(self._arms),
            "cameras": sorted(self._cameras),
            "bases": sorted(self._bases),
        }

    def rpc_world_play(self, ctx: Any) -> None:
        self.timeline.play()

    def rpc_world_pause(self, ctx: Any) -> None:
        self.timeline.pause()

    async def rpc_world_reset(self, ctx: Any) -> None:
        """Back to the authored state, then play."""
        self.timeline.stop()
        await _next_update()
        self.timeline.play()

    def rpc_world_add_usd(
        self,
        ctx: Any,
        usd_path: str,
        prim_path: str,
        position: Optional[List[float]] = None,
    ) -> None:
        from isaacsim.core.utils.stage import add_reference_to_stage

        _check_prim_path(prim_path)
        if _usd_exists(usd_path) is False:
            raise ValueError(f"usd not found: {usd_path}")
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        _set_pose(prim_path, position, None)

    def rpc_world_spawn_prop(self, ctx: Any, prop: Dict[str, Any]) -> Dict[str, Any]:
        """Add a configured prop unless a prim with its name already exists."""
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.utils.stage import add_reference_to_stage

        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = _prim_name(str(prop["name"]))
        prim_path = f"/World/{name}"
        if _prim_exists(prim_path):
            return {"prim_path": prim_path, "spawned": False}
        position = _vec3(prop.get("position"))
        kind = str(prop.get("type", "cube"))

        if kind == "usd":
            usd_path = prop.get("usd_path")
            if not usd_path:
                raise ValueError(f"prop {name}: type 'usd' needs usd_path")
            if _usd_exists(usd_path) is False:
                raise ValueError(f"prop {name}: usd not found: {usd_path}")
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            _set_pose(prim_path, position, None)
            return {"prim_path": prim_path, "spawned": True}

        if kind != "cube":
            raise ValueError(f"prop {name}: unknown type {kind!r} (cube or usd)")

        kwargs: Dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            position=np.array(position),
            size=float(prop.get("size", 0.05)),
        )
        if prop.get("scale") is not None:
            kwargs["scale"] = np.array([float(v) for v in prop["scale"]])
        if prop.get("color") is not None:
            kwargs["color"] = np.array([float(v) for v in prop["color"]])
        (FixedCuboid if prop.get("fixed") else DynamicCuboid)(**kwargs)
        return {"prim_path": prim_path, "spawned": True}

    # ------------------------------------------------------------------
    # arms
    # ------------------------------------------------------------------

    def rpc_arm_attach(
        self, ctx: Any, name: str, attrs: Dict[str, Any]
    ) -> Dict[str, Any]:
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim

        prim_path = _component_prim_path(name, attrs)
        ee_path = attrs.get("end_effector_prim")
        existing = self._arms.get(name)
        spawned = False
        if not _still_valid(existing, prim_path):
            spawned = self._spawn(prim_path, attrs)
            existing = None
        ee = None
        if ee_path:
            _require_prim(ee_path)
            ee = SingleXFormPrim(ee_path)
        if existing is not None:
            existing.ee = ee
        else:
            articulation = SingleArticulation(prim_path=prim_path, name=name)
            self._arms[name] = _Arm(
                prim_path, _stage_id(), _Articulated(self, prim_path, articulation), ee
            )
        return _attach_info(prim_path, spawned)

    async def rpc_arm_get_joint_positions(self, ctx: Any, name: str) -> List[float]:
        art = await self._lookup(self._arms, "arm", name).body.ready()
        return [float(v) for v in art.get_joint_positions()]

    async def rpc_arm_set_joint_targets(
        self, ctx: Any, name: str, positions: List[float]
    ) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        art = await self._lookup(self._arms, "arm", name).body.ready()
        if len(positions) != art.num_dof:
            raise ValueError(
                f"arm {name} has {art.num_dof} joints, got {len(positions)} positions"
            )
        art.apply_action(
            ArticulationAction(joint_positions=np.array(positions, dtype=float))
        )

    async def rpc_arm_stop(self, ctx: Any, name: str) -> None:
        """Hold the current position (a stopped simulation is already still)."""
        from isaacsim.core.utils.types import ArticulationAction

        arm = self._lookup(self._arms, "arm", name)
        if self.timeline.is_stopped():
            return
        art = await arm.body.ready()
        art.apply_action(ArticulationAction(joint_positions=art.get_joint_positions()))

    def rpc_arm_get_end_pose(self, ctx: Any, name: str) -> Dict[str, List[float]]:
        arm = self._lookup(self._arms, "arm", name)
        if arm.ee is None:
            raise NotImplementedError(
                "set end_effector_prim in the arm config to report end position"
            )
        pos, quat = arm.ee.get_world_pose()
        return {
            "position": [float(v) for v in pos[:3]],
            "orientation_wxyz": [float(v) for v in quat[:4]],
        }

    # ------------------------------------------------------------------
    # cameras
    # ------------------------------------------------------------------

    def rpc_camera_attach(
        self, ctx: Any, name: str, attrs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """attrs carry poses the module already resolved (see
        camera_placement in the module): spawn_pose goes to the Camera
        constructor, then local_pose (relative to parent_prim) or world_pose
        is applied - all only when the camera prim is created here."""
        from isaacsim.sensors.camera import Camera

        parent = attrs.get("parent_prim")
        if parent:
            _require_prim(parent)
            prim_path = f"{parent.rstrip('/')}/{_prim_name(name)}"
        else:
            prim_path = _component_prim_path(name, attrs)
        resolution = (int(attrs.get("width", 640)), int(attrs.get("height", 480)))
        fov = attrs.get("fov_deg")

        existing = self._cameras.get(name)
        if _still_valid(existing, prim_path):
            if existing.resolution != resolution:
                existing.camera.set_resolution(resolution)
                existing.resolution = resolution
            if fov:
                _set_fov(existing.camera, float(fov))
            return _attach_info(prim_path, False)

        created = not _prim_exists(prim_path)
        kwargs: Dict[str, Any] = dict(
            prim_path=prim_path, name=name, resolution=resolution
        )
        spawn = attrs.get("spawn_pose") or {}
        if created and spawn.get("position") is not None:
            kwargs["position"] = list(spawn["position"])
        if created and spawn.get("orientation_wxyz") is not None:
            kwargs["orientation"] = list(spawn["orientation_wxyz"])
        cam = Camera(**kwargs)
        cam.initialize()

        if created:
            local, world = attrs.get("local_pose"), attrs.get("world_pose")
            if local:
                cam.set_local_pose(
                    list(local["position"]),
                    list(local["orientation_wxyz"]),
                    camera_axes="usd",
                )
            elif world:
                cam.set_world_pose(
                    list(world["position"]),
                    list(world["orientation_wxyz"]),
                    camera_axes="world",
                )
            # new cameras default to a 70 degree horizontal FOV - the usd
            # default lens is ~24 degrees, which reads as "zoomed way in"
            _set_fov(cam, float(fov or 70.0))
        elif fov:
            _set_fov(cam, float(fov))

        self._cameras[name] = _Camera(prim_path, _stage_id(), cam, resolution)
        return _attach_info(prim_path, created)

    def rpc_camera_get_image(
        self, ctx: Any, name: str, mime_type: str = "image/jpeg"
    ) -> Binary:
        cam = self._lookup(self._cameras, "camera", name).camera
        frame = cam.get_rgba()
        if frame is None or frame.size == 0:
            raise NotReady("no frame available yet - is the simulation playing?")
        # copy now; the frame buffer belongs to the renderer
        rgb = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
        return Binary(encode=lambda: encode_rgb(rgb, mime_type))

    # ------------------------------------------------------------------
    # bases
    # ------------------------------------------------------------------

    def rpc_base_attach(
        self, ctx: Any, name: str, attrs: Dict[str, Any]
    ) -> Dict[str, Any]:
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot

        meta = _asset_meta(attrs)
        wheel_joints = attrs.get("wheel_joints") or meta.get("wheel_joints")
        if not wheel_joints or len(wheel_joints) != 2:
            raise ValueError(
                "base needs wheel_joints: [left_joint_name, right_joint_name] "
                "(known assets like 'jetbot' provide defaults)"
            )
        wheel_radius = float(attrs.get("wheel_radius", meta.get("wheel_radius", 0.05)))
        wheel_base = float(attrs.get("wheel_base", meta.get("wheel_base", 0.3)))
        controller = DifferentialController(
            name=f"{name}_controller", wheel_radius=wheel_radius, wheel_base=wheel_base
        )

        prim_path = _component_prim_path(name, attrs)
        existing = self._bases.get(name)
        spawned = False
        if _still_valid(existing, prim_path):
            existing.controller = controller
            existing.wheel_radius = wheel_radius
            existing.wheel_base = wheel_base
        else:
            spawned = self._spawn(prim_path, attrs)
            robot = WheeledRobot(
                prim_path=prim_path,
                name=name,
                wheel_dof_names=list(wheel_joints),
                create_robot=False,
            )
            self._bases[name] = _Base(
                prim_path,
                _stage_id(),
                _Articulated(self, prim_path, robot),
                controller,
                wheel_radius,
                wheel_base,
            )
        info = _attach_info(prim_path, spawned)
        info.update(wheel_radius=wheel_radius, wheel_base=wheel_base)
        return info

    async def rpc_base_set_velocity(
        self, ctx: Any, name: str, linear: float, angular: float
    ) -> None:
        base = self._lookup(self._bases, "base", name)
        await base.body.ready()
        base.cmd = (float(linear), float(angular))
        base.owner = ctx.id

    def rpc_base_stop(self, ctx: Any, name: str) -> None:
        self._lookup(self._bases, "base", name).cmd = (0.0, 0.0)

    def rpc_base_is_moving(self, ctx: Any, name: str) -> bool:
        return self._lookup(self._bases, "base", name).cmd != (0.0, 0.0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup(registry: Dict[str, Any], kind: str, name: str) -> Any:
        """The attached component, or NotAttached (which makes the module
        re-attach) if it never was or its prim is gone - a new stage was
        opened, or the prim was deleted."""
        entry = registry.get(name)
        if entry is None or not _still_valid(entry, entry.prim_path):
            registry.pop(name, None)
            raise NotAttached(f"{kind} {name!r} is not attached")
        return entry

    def _spawn(self, prim_path: str, attrs: Dict[str, Any]) -> bool:
        """Reference the configured usd at prim_path unless something is
        already there. Returns whether it spawned."""
        from isaacsim.core.utils.stage import add_reference_to_stage

        if _prim_exists(prim_path):
            return False
        usd = _resolve_usd(attrs)
        if not usd:
            raise ValueError(
                f"nothing at {prim_path} to attach to; set asset or usd_path to "
                "spawn one"
            )
        add_reference_to_stage(usd_path=usd, prim_path=prim_path)
        # authored on the USD prim, so it persists across timeline stop/play
        try:
            _set_pose(prim_path, attrs.get("position"), attrs.get("orientation_wxyz"))
        except Exception:
            LOGGER.exception("failed to set the spawn pose of %s", prim_path)
        LOGGER.info("spawned %s at %s", usd, prim_path)
        return True


async def _next_update() -> None:
    await omni.kit.app.get_app().next_update_async()


def _stage() -> Optional[Usd.Stage]:
    return omni.usd.get_context().get_stage()


def _stage_id() -> int:
    return int(omni.usd.get_context().get_stage_id())


def _check_prim_path(path: str) -> None:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"prim paths must be absolute, got {path!r}")


def _prim_exists(path: str) -> bool:
    _check_prim_path(path)
    stage = _stage()
    return bool(stage) and stage.GetPrimAtPath(path).IsValid()


def _still_valid(entry: Any, prim_path: str) -> bool:
    return (
        entry is not None
        and entry.prim_path == prim_path
        and entry.stage_id == _stage_id()
        and _prim_exists(prim_path)
    )


def _require_prim(prim_path: str) -> None:
    """Raise a helpful error if prim_path doesn't exist in the stage."""
    if _prim_exists(prim_path):
        return
    parent_path = prim_path.rsplit("/", 1)[0] or "/"
    hint = ""
    parent = _stage().GetPrimAtPath(parent_path)
    if parent.IsValid():
        children = [c.GetName() for c in parent.GetChildren()]
        hint = f"; children of {parent_path}: {children}"
    raise ValueError(f"prim not found: {prim_path}{hint}")


def _prim_name(name: str) -> str:
    """Component names may contain characters USD prim names can't."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _component_prim_path(name: str, attrs: Dict[str, Any]) -> str:
    path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
    _check_prim_path(path)
    return path


def _vec3(value: Any) -> List[float]:
    if value is None:
        return [0.0, 0.0, 0.0]
    vec = [float(v) for v in value]
    if len(vec) != 3:
        raise ValueError(f"expected [x, y, z], got {value!r}")
    return vec


def _set_pose(
    prim_path: str, position: Any, orientation_wxyz: Optional[List[float]]
) -> None:
    from isaacsim.core.prims import SingleXFormPrim

    kwargs: Dict[str, Any] = {"position": _vec3(position)}
    if orientation_wxyz is not None:
        kwargs["orientation"] = [float(v) for v in orientation_wxyz]
    SingleXFormPrim(prim_path).set_world_pose(**kwargs)


def _attach_info(prim_path: str, spawned: bool) -> Dict[str, Any]:
    """What the module gets back from an attach: where the prim is, as
    authored in USD, so it can flag disagreement with viam's frame config."""
    prim = _stage().GetPrimAtPath(prim_path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform()
    transform.SetMatrix(matrix)
    t = transform.GetTranslation()
    q = transform.GetRotation().GetQuat()
    im = q.GetImaginary()
    return {
        "prim_path": prim_path,
        "spawned": spawned,
        "position": [float(t[0]), float(t[1]), float(t[2])],
        "orientation_wxyz": [
            float(q.GetReal()),
            float(im[0]),
            float(im[1]),
            float(im[2]),
        ],
    }


def _set_fov(cam: Any, fov_deg: float) -> None:
    # via the aperture so usd unit conventions cancel out
    import math

    aperture = cam.get_horizontal_aperture()
    cam.set_focal_length(aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0)))


def _usd_exists(path: str) -> Optional[bool]:
    """True/False if we can check, None if omni.client is unavailable."""
    try:
        import omni.client
    except ImportError:
        return None
    try:
        result, _ = omni.client.stat(path)
        return result == omni.client.Result.OK
    except Exception:
        return None


def _asset_meta(attrs: Dict[str, Any]) -> Dict[str, Any]:
    asset = attrs.get("asset")
    if not asset:
        return {}
    if asset not in KNOWN_ASSETS:
        raise ValueError(
            f"unknown asset {asset!r}; known: {sorted(KNOWN_ASSETS)} "
            "(or set usd_path directly)"
        )
    return KNOWN_ASSETS[asset]


def _resolve_usd(attrs: Dict[str, Any]) -> Optional[str]:
    """Absolute usd path to spawn, or None if the config names none."""
    usd = attrs.get("usd_path")
    meta = _asset_meta(attrs)
    if meta and not usd:
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
        if root is None:
            raise RuntimeError("could not reach the isaac sim assets server")
        candidates = meta["usd"]
        for rel in candidates:
            if _usd_exists(root + rel) is not False:
                usd = root + rel
                break
        if usd is None:
            raise ValueError(
                f"asset {attrs['asset']!r}: none of {candidates} exist under {root}; "
                "the asset layout may have changed in this isaac release"
            )
    # a USD reference to a missing file "succeeds" but leaves an empty prim,
    # which later fails with confusing physics-tensor errors - catch it here
    if usd and _usd_exists(usd) is False:
        raise ValueError(f"usd not found: {usd}")
    return usd
