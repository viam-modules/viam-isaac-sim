"""The singleton that owns Isaac Sim.

Isaac Sim (Omniverse Kit) wants to be created and stepped from a single
thread, so the module runs it on the process main thread (see main.py) and
everything else - the Viam module server, component handlers - submits work
to that thread through a queue. Handles returned by create_arm/create_camera/
create_base wrap that queue so component models can stay simple.

A "mock" backend (world attribute: {"mock": true}) implements the same
handle interfaces with plain python so the module can run and be tested on
machines without Isaac Sim installed.
"""

import math
import queue
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from viam.logging import getLogger

LOGGER = getLogger("viam-isaac-sim")

# Assets shipped on the Isaac Sim nucleus/content server, addressable by a
# short name in component config. Paths are relative to the assets root;
# where isaac 5.0 moved an asset, the 5.0 path is listed first with the 4.x
# path as a fallback - the first candidate that exists is used.
_UR_KINEMATICS = "https://raw.githubusercontent.com/viam-modules/universal-robots/main/src/kinematics"

KNOWN_ASSETS: Dict[str, Dict[str, Any]] = {
    "ur3e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur3e.json",
    },
    "ur5e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur5e.json",
    },
    "ur10": {"usd": ["/Isaac/Robots/UniversalRobots/ur10/ur10.usd"]},
    "ur10e": {"usd": ["/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"]},
    "ur16e": {"usd": ["/Isaac/Robots/UniversalRobots/ur16e/ur16e.usd"]},
    "ur20": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur20/ur20.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur20.json",
    },
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


@dataclass
class SimConfig:
    mock: bool = False
    headless: bool = True
    livestream: bool = True
    usd_stage: Optional[str] = None
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    boot_timeout: float = 300.0
    # IP the livestream advertises to clients; auto-detected if empty
    livestream_public_ip: str = ""
    # resolution kit renders (and therefore streams) at
    livestream_width: int = 1280
    livestream_height: int = 720
    # props to spawn into the scene at boot; each entry:
    #   {"type": "cube"|"usd", "name": ..., "position": [x,y,z] (m),
    #    "size": edge_m, "scale": [sx,sy,sz], "color": [r,g,b] 0-1,
    #    "fixed": bool, "usd_path": ...}
    props: List[Dict[str, Any]] = field(default_factory=list)
    # kit console verbosity (verbose/info/warning/error). Kit prints thousands
    # of lines at info, and viam-server records the module's stderr as
    # error-level logs, so default to warning.
    kit_log_level: str = "warning"
    # add_default_ground_plane loads the Grid env USD, which bundles floor
    # collider AND the lighting rig; when false, we add fallback lights
    isaac_default_environment: bool = True


class SimManager:
    """Owns the sim thread. Get the process-wide instance via SimManager.get()."""

    _instance: Optional["SimManager"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "SimManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SimManager()
            return cls._instance

    def __init__(self) -> None:
        self._tasks: "queue.Queue[Tuple[Callable[[], Any], Future]]" = queue.Queue()
        self._boot_requested = threading.Event()
        self._booted = threading.Event()
        self._boot_error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._sim_thread_id: Optional[int] = None

        self.cfg: Optional[SimConfig] = None
        self.mock = False
        self._sim_app = None
        self.world = None
        self._isaac = None  # lazily-populated namespace of isaac imports
        self._step_callbacks: Dict[str, Callable[[float], None]] = {}
        # component name -> (spawn attrs, handle). viam-server rebuilds
        # resources on config change, but prims can't be re-spawned without
        # restarting kit, so handles are cached per component name.
        self._handles: Dict[str, Tuple[Dict[str, Any], Any]] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def ensure_booted(self, cfg: SimConfig) -> None:
        """Called by the world component's reconfigure. Boots the sim on the
        sim thread the first time; subsequent calls with a different config
        log that a module restart is required (Kit can't be re-created)."""
        if self._booted.is_set():
            if self.cfg != cfg:
                LOGGER.warning(
                    "isaac sim is already running; changes to world config "
                    "(stage/headless/etc) require restarting the module"
                )
            return
        if self._boot_error is not None:
            raise RuntimeError(f"isaac sim failed to boot previously: {self._boot_error}")

        self.cfg = cfg
        self._boot_requested.set()
        if not self._booted.wait(timeout=cfg.boot_timeout):
            raise TimeoutError(f"isaac sim did not boot within {cfg.boot_timeout}s")
        if self._boot_error is not None:
            raise RuntimeError(f"isaac sim failed to boot: {self._boot_error}")

    def request_stop(self) -> None:
        self._stop.set()

    def main_loop(self) -> None:
        """Run forever on the owning (main) thread: wait for a boot request,
        boot, then step the sim while draining queued tasks."""
        self._sim_thread_id = threading.get_ident()

        while not self._stop.is_set() and not self._boot_requested.wait(timeout=0.1):
            self._drain_tasks()

        if self._stop.is_set():
            return

        try:
            self._boot()
        except BaseException as e:  # SimulationApp failures can be SystemExit
            LOGGER.exception("failed to boot isaac sim")
            self._boot_error = e
            self._booted.set()
            return
        self._booted.set()

        last = time.monotonic()
        while not self._stop.is_set():
            self._drain_tasks()
            now = time.monotonic()
            dt = now - last
            last = now
            if self.mock:
                for cb in list(self._step_callbacks.values()):
                    cb(dt)
                time.sleep(0.01)
            else:
                self.world.step(render=True)

        if self._sim_app is not None:
            try:
                self._sim_app.close()
            except Exception:
                LOGGER.exception("error closing isaac sim")

    def _drain_tasks(self) -> None:
        while True:
            try:
                fn, fut = self._tasks.get_nowait()
            except queue.Empty:
                return
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as e:
                    fut.set_exception(e)

    def run(self, fn: Callable[[], Any], timeout: float = 30.0) -> Any:
        """Run fn on the sim thread and return its result."""
        if threading.get_ident() == self._sim_thread_id:
            return fn()
        fut: Future = Future()
        self._tasks.put((fn, fut))
        return fut.result(timeout=timeout)

    # ------------------------------------------------------------------
    # boot
    # ------------------------------------------------------------------

    def _boot(self) -> None:
        cfg = self.cfg
        assert cfg is not None
        if cfg.mock:
            LOGGER.info("booting in MOCK mode - no isaac sim")
            self.mock = True
            return

        LOGGER.info("booting isaac sim (headless=%s)...", cfg.headless)
        try:
            from isaacsim import SimulationApp  # isaac sim >= 4.5
        except ImportError:
            from omni.isaac.kit import SimulationApp  # older releases

        # quiet kit's console stream; unknown argv entries are forwarded to kit
        level = cfg.kit_log_level.capitalize()
        sys.argv.append(f"--/log/outputStreamLevel={level}")

        streaming = cfg.livestream and cfg.headless
        launch: Dict[str, Any] = {"headless": cfg.headless}
        if streaming:
            # SimulationApp hides kit's UI whenever headless is set, which
            # leaves a connected streaming client looking at an empty frame.
            # These mirror isaacsim.exp.full.streaming and the livestream
            # standalone example: keep the UI, render it at a size worth
            # streaming, and show the default grid.
            launch.update(
                hide_ui=False,
                width=cfg.livestream_width,
                height=cfg.livestream_height,
                window_width=cfg.livestream_width,
                window_height=cfg.livestream_height,
                renderer="RaytracedLighting",
                display_options=3286,
            )

        self._sim_app = SimulationApp(launch)

        try:
            import carb.settings

            carb.settings.get_settings().set("/log/outputStreamLevel", level)
        except Exception:
            pass

        if streaming:
            try:
                try:
                    from isaacsim.core.utils.extensions import enable_extension
                except ImportError:
                    from omni.isaac.core.utils.extensions import enable_extension

                ip = cfg.livestream_public_ip or _local_ip()
                self._sim_app.set_setting("/app/window/drawMouse", True)
                # the client asks to resize the stream as soon as it connects;
                # without this the request is refused and the view stays blank
                self._sim_app.set_setting("/app/livestream/allowResize", True)
                self._sim_app.set_setting("/app/livestream/port", 49100)
                if ip:
                    self._sim_app.set_setting("/app/livestream/publicEndpointAddress", ip)
                # 5.0 ships the streaming service under omni.services; 4.5 and
                # older only have the kit extension
                for ext in ("omni.services.livestream.nvcf", "omni.kit.livestream.webrtc"):
                    if enable_extension(ext):
                        break
                else:
                    raise RuntimeError("no livestream extension could be enabled")
                LOGGER.info(
                    "livestream enabled - connect the 'Isaac Sim WebRTC Streaming "
                    "Client' app to %s (TCP 49100 + UDP 47998 must be reachable)",
                    ip or "<this machine's IP>",
                )
            except Exception:
                LOGGER.exception("could not enable livestream; continuing without it")

        self._isaac = _import_isaac()

        if cfg.usd_stage:
            LOGGER.info("opening stage %s", cfg.usd_stage)
            self._isaac.open_stage(cfg.usd_stage)

        self.world = self._isaac.World(
            physics_dt=cfg.physics_dt,
            rendering_dt=cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )
        if not cfg.usd_stage:
            if cfg.isaac_default_environment:
                self.world.scene.add_default_ground_plane()
            else:
                self._add_fallback_lighting()
        for prop in cfg.props:
            try:
                self._spawn_prop(prop)
            except Exception:
                LOGGER.exception("failed to spawn prop %s", prop.get("name"))
        self.world.reset()
        LOGGER.info("isaac sim world ready")

    def _add_fallback_lighting(self) -> None:
        from pxr import Sdf, UsdLux

        stage = self.world.stage
        distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DefaultDistantLight"))
        distant.CreateIntensityAttr(3000.0)
        distant.CreateAngleAttr(0.53)
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DefaultDomeLight"))
        dome.CreateIntensityAttr(500.0)
        LOGGER.info("added fallback DistantLight + DomeLight")

    def _spawn_prop(self, prop: Dict[str, Any]) -> None:
        """Add a configured prop to the scene (runs on the sim thread,
        before the initial world.reset)."""
        import numpy as np

        from .spatial import to_vec3

        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = _prim_name(str(prop["name"]))
        prim_path = f"/World/{name}"
        position = list(to_vec3(prop.get("position")))
        kind = str(prop.get("type", "cube"))

        if kind == "usd":
            usd_path = prop.get("usd_path")
            if not usd_path:
                raise ValueError(f"prop {name}: type 'usd' needs usd_path")
            if self._usd_exists(usd_path) is False:
                raise ValueError(f"prop {name}: usd not found: {usd_path}")
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            self._isaac.SingleXFormPrim(prim_path).set_world_pose(position=position)
            return

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
        cls = self._isaac.FixedCuboid if prop.get("fixed") else self._isaac.DynamicCuboid
        self.world.scene.add(cls(**kwargs))

    def _require_booted(self) -> None:
        if not self._booted.is_set():
            raise RuntimeError(
                "isaac sim world is not running - configure a "
                "viam-labs:isaac-sim:world component and depend on it"
            )
        if self._boot_error is not None:
            raise RuntimeError(f"isaac sim failed to boot: {self._boot_error}")

    # ------------------------------------------------------------------
    # world controls (used by the world component's DoCommand)
    # ------------------------------------------------------------------

    def play(self) -> None:
        self._require_booted()
        if not self.mock:
            self.run(lambda: self.world.play())

    def pause(self) -> None:
        self._require_booted()
        if not self.mock:
            self.run(lambda: self.world.pause())

    def reset(self) -> None:
        self._require_booted()
        if not self.mock:
            self.run(lambda: self.world.reset())

    def status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "booted": self._booted.is_set(),
            "mock": self.mock,
            "error": str(self._boot_error) if self._boot_error else "",
        }
        if self._booted.is_set() and not self.mock:
            out["playing"] = self.run(lambda: bool(self.world.is_playing()))
            out["sim_time"] = self.run(lambda: float(self.world.current_time))
        return out

    def add_usd_reference(
        self,
        usd_path: str,
        prim_path: str,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._require_booted()
        if self.mock:
            return

        def _add():
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            prim = self._isaac.SingleXFormPrim(prim_path)
            prim.set_world_pose(position=list(position))

        self.run(_add, timeout=60.0)

    # ------------------------------------------------------------------
    # component factories
    # ------------------------------------------------------------------

    # attributes that only affect the viam-side model, not the spawned prim
    _RUNTIME_KEYS = frozenset(
        {"world", "move_timeout_sec", "max_linear_mps", "max_angular_rps"}
    )

    def _cached_handle(
        self, kind: str, name: str, attrs: Dict[str, Any], factory: Callable[[], Any]
    ) -> Any:
        if name in self._handles:
            old_attrs, handle = self._handles[name]
            strip = lambda a: {k: v for k, v in a.items() if k not in self._RUNTIME_KEYS}
            if strip(old_attrs) != strip(attrs):
                LOGGER.warning(
                    "%s %r: spawn config changed but the prim is already in the "
                    "stage; restart the module to apply",
                    kind,
                    name,
                )
            return handle
        handle = factory()
        self._handles[name] = (dict(attrs), handle)
        return handle

    def _usd_exists(self, path: str) -> Optional[bool]:
        """True/False if we can check, None if omni.client is unavailable."""
        client = getattr(self._isaac, "client", None)
        if client is None:
            return None
        try:
            result, _ = client.stat(path)
            return result == client.Result.OK
        except Exception:
            return None

    def _resolve_usd(self, attrs: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Return (absolute usd path or None, known-asset metadata)."""
        meta: Dict[str, Any] = {}
        usd = attrs.get("usd_path")
        asset = attrs.get("asset")
        if asset:
            if asset not in KNOWN_ASSETS:
                raise ValueError(
                    f"unknown asset {asset!r}; known: {sorted(KNOWN_ASSETS)} "
                    "(or set usd_path directly)"
                )
            meta = KNOWN_ASSETS[asset]
            if not usd:
                root = self._isaac.get_assets_root_path()
                if root is None:
                    raise RuntimeError("could not reach the isaac sim assets server")
                candidates = meta["usd"]
                for rel in candidates:
                    if self._usd_exists(root + rel) is not False:
                        usd = root + rel
                        break
                if usd is None:
                    raise ValueError(
                        f"asset {asset!r}: none of {candidates} exist under {root}; "
                        "the asset layout may have changed in this isaac release"
                    )
        # a USD reference to a missing file "succeeds" but leaves an empty
        # prim, which later fails with confusing physics-tensor errors -
        # catch it here instead
        if usd and self._usd_exists(usd) is False:
            raise ValueError(f"usd not found: {usd}")
        return usd, meta

    def create_arm(self, name: str, attrs: Dict[str, Any]) -> "ArmHandle":
        self._require_booted()
        if self.mock:
            factory = lambda: MockArmHandle(name, attrs)
        else:
            factory = lambda: self.run(
                lambda: self._create_arm_isaac(name, attrs), timeout=120.0
            )
        return self._cached_handle("arm", name, attrs, factory)

    def _create_arm_isaac(self, name: str, attrs: Dict[str, Any]) -> "IsaacArmHandle":
        from .spatial import to_vec3

        usd, _ = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        if usd:
            self._isaac.add_reference_to_stage(usd_path=usd, prim_path=prim_path)

        # Set the base pose directly on the USD prim. The `position=` /
        # `orientation=` kwargs on SingleArticulation don't persist across the
        # world.reset() below when another articulation is already in the
        # scene, so every arm silently ends up at the world origin and their
        # bodies penetrate. Writing to USD persists across resets.
        position = to_vec3(attrs.get("position"))
        pose_kwargs: Dict[str, Any] = {"position": list(position)}
        if attrs.get("orientation_wxyz") is not None:
            pose_kwargs["orientation"] = [float(v) for v in attrs["orientation_wxyz"]]
        try:
            self._isaac.SingleXFormPrim(prim_path).set_world_pose(**pose_kwargs)
        except Exception:
            LOGGER.exception("failed to set base pose for %s", name)

        art = self._isaac.SingleArticulation(prim_path=prim_path, name=name)
        self.world.scene.add(art)
        self.world.reset()

        ee = None
        ee_path = attrs.get("end_effector_prim")
        if ee_path:
            ee = self._isaac.SingleXFormPrim(ee_path)
        return IsaacArmHandle(self, art, ee)

    def create_camera(self, name: str, attrs: Dict[str, Any]) -> "CameraHandle":
        self._require_booted()
        if self.mock:
            factory = lambda: MockCameraHandle(name, attrs)
        else:
            factory = lambda: self.run(
                lambda: self._create_camera_isaac(name, attrs), timeout=120.0
            )
        return self._cached_handle("camera", name, attrs, factory)

    def _create_camera_isaac(self, name: str, attrs: Dict[str, Any]) -> "IsaacCameraHandle":
        import math

        from .spatial import quat_from_euler_deg, to_vec3

        parent = attrs.get("parent_prim")
        if parent:
            self._require_prim(parent)
            prim_path = f"{parent.rstrip('/')}/{_prim_name(name)}"
        else:
            prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        width = int(attrs.get("width", 640))
        height = int(attrs.get("height", 480))

        kwargs: Dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            resolution=(width, height),
        )
        if attrs.get("position") is not None:
            kwargs["position"] = list(to_vec3(attrs.get("position")))
        if attrs.get("orientation_rpy_deg") is not None:
            r, p, y = to_vec3(attrs.get("orientation_rpy_deg"))
            kwargs["orientation"] = list(quat_from_euler_deg(r, p, y))

        cam = self._isaac.Camera(**kwargs)
        cam.initialize()

        if parent:
            # camera rides a (possibly moving) link; pose is local to it.
            # default: small standoff along the link, looking out the +Z
            # (tool) axis - 180deg about X flips the usd camera's -Z forward.
            local_pos = to_vec3(attrs.get("local_position"), default=(0.0, 0.0, 0.05))
            r, p, y = to_vec3(
                attrs.get("local_orientation_rpy_deg"), default=(180.0, 0.0, 0.0)
            )
            quat = quat_from_euler_deg(r, p, y)
            cam.set_local_pose(list(local_pos), list(quat), camera_axes="usd")
        elif attrs.get("target") is not None:
            # aim at a target point (world axes: +X forward, +Z up)
            from .spatial import look_at_quat

            position = to_vec3(attrs.get("position"), default=(3.0, 3.0, 2.5))
            quat = look_at_quat(position, to_vec3(attrs.get("target")))
            cam.set_world_pose(list(position), list(quat), camera_axes="world")
        elif attrs.get("orientation_wxyz") is not None:
            position = to_vec3(attrs.get("position"))
            quat = tuple(float(v) for v in attrs["orientation_wxyz"])
            cam.set_world_pose(list(position), list(quat), camera_axes="world")

        # newly created cameras default to a 70 degree horizontal FOV - the
        # usd default lens is ~24 degrees, which reads as "zoomed way in".
        # computed via the aperture so usd unit conventions cancel out.
        if not attrs.get("prim_path"):
            fov = float(attrs.get("fov_deg", 70.0))
            aperture = cam.get_horizontal_aperture()
            cam.set_focal_length(aperture / (2.0 * math.tan(math.radians(fov) / 2.0)))
        elif attrs.get("fov_deg"):
            fov = float(attrs["fov_deg"])
            aperture = cam.get_horizontal_aperture()
            cam.set_focal_length(aperture / (2.0 * math.tan(math.radians(fov) / 2.0)))
        return IsaacCameraHandle(self, cam)

    def _require_prim(self, prim_path: str) -> None:
        """Raise a helpful error if prim_path doesn't exist in the stage."""
        get_prim = getattr(self._isaac, "get_prim_at_path", None)
        if get_prim is None:
            return
        prim = get_prim(prim_path)
        if prim is None or not prim.IsValid():
            parent_path = prim_path.rsplit("/", 1)[0] or "/"
            hint = ""
            parent = get_prim(parent_path)
            if parent is not None and parent.IsValid():
                children = [c.GetName() for c in parent.GetChildren()]
                hint = f"; children of {parent_path}: {children}"
            raise ValueError(f"prim not found: {prim_path}{hint}")

    def create_base(self, name: str, attrs: Dict[str, Any]) -> "BaseHandle":
        self._require_booted()
        if self.mock:
            factory = lambda: MockBaseHandle(name, attrs)
        else:
            factory = lambda: self.run(
                lambda: self._create_base_isaac(name, attrs), timeout=120.0
            )
        return self._cached_handle("base", name, attrs, factory)

    def _create_base_isaac(self, name: str, attrs: Dict[str, Any]) -> "IsaacBaseHandle":
        from .spatial import to_vec3

        usd, meta = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        wheel_joints = attrs.get("wheel_joints") or meta.get("wheel_joints")
        if not wheel_joints or len(wheel_joints) != 2:
            raise ValueError(
                "base needs wheel_joints: [left_joint_name, right_joint_name] "
                "(known assets like 'jetbot' provide defaults)"
            )
        wheel_radius = float(attrs.get("wheel_radius", meta.get("wheel_radius", 0.05)))
        wheel_base = float(attrs.get("wheel_base", meta.get("wheel_base", 0.3)))
        position = to_vec3(attrs.get("position"))

        base_kwargs: Dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            wheel_dof_names=list(wheel_joints),
            create_robot=usd is not None,
            usd_path=usd,
            position=list(position),
        )
        if attrs.get("orientation_wxyz") is not None:
            base_kwargs["orientation"] = [float(v) for v in attrs["orientation_wxyz"]]
        robot = self._isaac.WheeledRobot(**base_kwargs)
        self.world.scene.add(robot)
        self.world.reset()

        controller = self._isaac.DifferentialController(
            name=f"{name}_controller",
            wheel_radius=wheel_radius,
            wheel_base=wheel_base,
        )
        handle = IsaacBaseHandle(self, robot, controller, wheel_radius, wheel_base)
        self.world.add_physics_callback(f"{name}_drive", handle._on_physics_step)
        return handle


def _local_ip() -> str:
    """Best-effort primary local IP (no traffic is actually sent)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return ""


def _prim_name(name: str) -> str:
    """Component names may contain characters USD prim names can't."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _import_isaac():
    """Import everything we need from isaac sim, tolerating the module
    renames across releases (isaacsim.* in >=4.5, omni.isaac.* before)."""

    class NS:
        pass

    ns = NS()

    try:
        from isaacsim.core.api import World
    except ImportError:
        from omni.isaac.core import World
    ns.World = World

    try:
        from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage
    ns.add_reference_to_stage = add_reference_to_stage
    ns.open_stage = open_stage

    try:
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:
        from omni.isaac.core.utils.nucleus import get_assets_root_path
    ns.get_assets_root_path = get_assets_root_path

    try:
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    except ImportError:
        from omni.isaac.core.articulations import Articulation as SingleArticulation
        from omni.isaac.core.prims import XFormPrim as SingleXFormPrim
    ns.SingleArticulation = SingleArticulation
    ns.SingleXFormPrim = SingleXFormPrim

    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    ns.ArticulationAction = ArticulationAction

    try:
        import omni.client
        ns.client = omni.client
    except ImportError:
        ns.client = None

    try:
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    except ImportError:
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
    ns.DynamicCuboid = DynamicCuboid
    ns.FixedCuboid = FixedCuboid

    try:
        from isaacsim.core.utils.prims import get_prim_at_path
    except ImportError:
        try:
            from omni.isaac.core.utils.prims import get_prim_at_path
        except ImportError:
            get_prim_at_path = None
    ns.get_prim_at_path = get_prim_at_path

    try:
        from isaacsim.sensors.camera import Camera
    except ImportError:
        from omni.isaac.sensor import Camera
    ns.Camera = Camera

    try:
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
    except ImportError:
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
    ns.WheeledRobot = WheeledRobot
    ns.DifferentialController = DifferentialController

    return ns


# ======================================================================
# Handles - the interface component models talk to. All public methods are
# safe to call from any thread.
# ======================================================================


class ArmHandle:
    def get_joint_positions(self) -> List[float]:  # radians
        raise NotImplementedError

    def set_joint_targets(self, positions: List[float]) -> None:
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def get_end_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """((x,y,z) meters, (w,x,y,z) quaternion) of the end effector."""
        raise NotImplementedError


class IsaacArmHandle(ArmHandle):
    def __init__(self, sim: SimManager, articulation, ee_prim) -> None:
        self._sim = sim
        self._art = articulation
        self._ee = ee_prim

    def get_joint_positions(self) -> List[float]:
        return self._sim.run(lambda: [float(v) for v in self._art.get_joint_positions()])

    def set_joint_targets(self, positions: List[float]) -> None:
        import numpy as np

        def _apply():
            action = self._sim._isaac.ArticulationAction(
                joint_positions=np.array(positions, dtype=float)
            )
            self._art.apply_action(action)

        self._sim.run(_apply)

    def is_moving(self) -> bool:
        def _check():
            vels = self._art.get_joint_velocities()
            if vels is None:
                return False
            return bool(max(abs(float(v)) for v in vels) > 1e-2)

        return self._sim.run(_check)

    def stop(self) -> None:
        # hold the current position
        current = self.get_joint_positions()
        self.set_joint_targets(current)

    def get_end_pose(self):
        if self._ee is None:
            raise NotImplementedError(
                "set end_effector_prim in the arm config to report end position"
            )

        def _pose():
            pos, quat = self._ee.get_world_pose()
            return (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )

        return self._sim.run(_pose)


class MockArmHandle(ArmHandle):
    """Joints move linearly toward their targets at a fixed speed."""

    SPEED = 1.0  # rad/s per joint

    def __init__(self, name: str, attrs: Dict[str, Any]) -> None:
        self.name = name
        dof = int(attrs.get("mock_dof", 6))
        self._lock = threading.Lock()
        self._start = [0.0] * dof
        self._target = [0.0] * dof
        self._t0 = time.monotonic()

    def _positions_at(self, now: float) -> List[float]:
        out = []
        dt = max(0.0, now - self._t0)
        for s, t in zip(self._start, self._target):
            delta = t - s
            travel = self.SPEED * dt
            if abs(delta) <= travel:
                out.append(t)
            else:
                out.append(s + math.copysign(travel, delta))
        return out

    def get_joint_positions(self) -> List[float]:
        with self._lock:
            return self._positions_at(time.monotonic())

    def set_joint_targets(self, positions: List[float]) -> None:
        with self._lock:
            now = time.monotonic()
            self._start = self._positions_at(now)
            if len(positions) != len(self._start):
                raise ValueError(
                    f"expected {len(self._start)} joint positions, got {len(positions)}"
                )
            self._target = list(positions)
            self._t0 = now

    def is_moving(self) -> bool:
        with self._lock:
            pos = self._positions_at(time.monotonic())
            return any(abs(p - t) > 1e-9 for p, t in zip(pos, self._target))

    def stop(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._start = self._positions_at(now)
            self._target = list(self._start)
            self._t0 = now

    def get_end_pose(self):
        # a fixed, deterministic pose for testing
        return ((0.3, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0))


class CameraHandle:
    def get_rgb(self):
        """Return an (H, W, 3) uint8 numpy array."""
        raise NotImplementedError


class IsaacCameraHandle(CameraHandle):
    def __init__(self, sim: SimManager, camera) -> None:
        self._sim = sim
        self._cam = camera

    def get_rgb(self):
        def _grab():
            frame = self._cam.get_rgba()
            if frame is None or frame.size == 0:
                raise RuntimeError(
                    "no frame available yet - is the simulation playing?"
                )
            return frame[:, :, :3].copy()

        return self._sim.run(_grab)


class MockCameraHandle(CameraHandle):
    def __init__(self, name: str, attrs: Dict[str, Any]) -> None:
        self.name = name
        self._w = int(attrs.get("width", 640))
        self._h = int(attrs.get("height", 480))

    def get_rgb(self):
        import numpy as np

        # gradient background with a time-based moving bar so images change
        img = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, self._w, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(0, 255, self._h, dtype=np.uint8)[:, None]
        x = int((time.monotonic() * 60) % self._w)
        img[:, max(0, x - 5) : x + 5, :] = 255
        return img


class BaseHandle:
    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError


class IsaacBaseHandle(BaseHandle):
    def __init__(self, sim: SimManager, robot, controller, wheel_radius: float, wheel_base: float) -> None:
        self._sim = sim
        self._robot = robot
        self._controller = controller
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self._cmd = (0.0, 0.0)
        self._lock = threading.Lock()

    def _on_physics_step(self, step_size: float) -> None:
        # runs on the sim thread every physics step
        with self._lock:
            lin, ang = self._cmd
        try:
            self._robot.apply_wheel_actions(self._controller.forward(command=[lin, ang]))
        except Exception:
            LOGGER.exception("error driving base")

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        with self._lock:
            self._cmd = (float(linear_mps), float(angular_rps))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def is_moving(self) -> bool:
        with self._lock:
            return self._cmd != (0.0, 0.0)


class MockBaseHandle(BaseHandle):
    def __init__(self, name: str, attrs: Dict[str, Any]) -> None:
        self.name = name
        self.wheel_radius = float(attrs.get("wheel_radius", 0.05))
        self.wheel_base = float(attrs.get("wheel_base", 0.3))
        self._cmd = (0.0, 0.0)
        self._lock = threading.Lock()

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        with self._lock:
            self._cmd = (float(linear_mps), float(angular_rps))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def is_moving(self) -> bool:
        with self._lock:
            return self._cmd != (0.0, 0.0)
