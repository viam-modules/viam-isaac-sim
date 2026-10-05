"""The singleton that links the module to Isaac Sim.

Isaac Sim runs as its own process with the viam_isaac_server Kit extension
enabled (see exts/). The world component points the module at that
extension, and the handles returned by create_arm/create_camera/create_base
turn component calls into requests to it, so component models can stay
simple.

A "mock" backend (world attribute: {"mock": true}) implements the same
handle interfaces with plain python so the module can run and be tested on
machines without Isaac Sim.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from viam.logging import getLogger

from .client import NotAttachedError, SimClient, SimError
from .images import encode_rgb
from .protocol import DEFAULT_PORT, MIME_RAW_RGB
from .spatial import look_at_quat, quat_from_euler_deg, to_vec3

LOGGER = getLogger("viam-isaac-sim")

DEFAULT_ADDRESS = f"localhost:{DEFAULT_PORT}"
CALL_TIMEOUT = 30.0
# attaching may spawn a USD fetched from the content server
ATTACH_TIMEOUT = 120.0

# official viam kinematics for known assets (the extension resolves which
# USD each name spawns)
_UR_KINEMATICS = "https://raw.githubusercontent.com/viam-modules/universal-robots/main/src/kinematics"

KNOWN_KINEMATICS: Dict[str, str] = {
    "ur3e": f"{_UR_KINEMATICS}/ur3e.json",
    "ur5e": f"{_UR_KINEMATICS}/ur5e.json",
    "ur20": f"{_UR_KINEMATICS}/ur20.json",
}


@dataclass
class SimConfig:
    mock: bool = False
    # host:port of the viam_isaac_server extension
    address: str = DEFAULT_ADDRESS
    connect_timeout: float = 10.0
    # props to spawn into the scene unless a prim with that name exists; each:
    #   {"type": "cube"|"usd", "name": ..., "position": [x,y,z] (m),
    #    "size": edge_m, "scale": [sx,sy,sz], "color": [r,g,b] 0-1,
    #    "fixed": bool, "usd_path": ...}
    props: List[Dict[str, Any]] = field(default_factory=list)


class SimManager:
    """Owns the connection to the extension. Get the process-wide instance
    via SimManager.get()."""

    _instance: Optional["SimManager"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "SimManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SimManager()
            return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cfg: Optional[SimConfig] = None
        self._client: Optional[SimClient] = None
        # component name -> handle. Mock state lives as long as the process,
        # the way prims outlive a reconfigure in a real stage.
        self._mock_handles: Dict[str, Any] = {}

    @property
    def mock(self) -> bool:
        return self.cfg is not None and self.cfg.mock

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def configure(self, cfg: SimConfig) -> None:
        """Called by the world component's reconfigure: connects to the
        extension (reconnecting if the address changed) and makes sure the
        configured props exist."""
        with self._lock:
            old = self._client
            if cfg.mock:
                self._client = None
            elif old is None or old.address != cfg.address:
                self._client = SimClient(cfg.address, on_connect=self._on_connect)
            if old is not None and old is not self._client:
                old.close()
            if self._client is not None:
                self._client.connect_timeout = cfg.connect_timeout
            self.cfg = cfg
            client = self._client

        if client is None:
            LOGGER.info("running in MOCK mode - no isaac sim")
        elif client.connected:
            self._on_connect(client)  # props may have changed
        else:
            client.connect()  # runs _on_connect

    def _on_connect(self, client: SimClient) -> None:
        """Runs on every (re)connect - e.g. after Isaac Sim restarts."""
        cfg = self.cfg
        if cfg is None or cfg.mock:
            return
        for prop in cfg.props:
            try:
                client.call("world.spawn_prop", {"prop": prop}, timeout=ATTACH_TIMEOUT)
            except (SimError, ValueError, TimeoutError):
                LOGGER.exception("failed to spawn prop %s", prop.get("name"))
        status = client.call("world.status")
        meters_per_unit = status.get("meters_per_unit", 1.0)
        if meters_per_unit != 1.0:
            LOGGER.warning(
                "the isaac sim stage is in units of %sm, but this module "
                "assumes meters; poses will be off by that factor",
                meters_per_unit,
            )

    def close(self) -> None:
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()

    def _require_configured(self) -> SimConfig:
        if self.cfg is None:
            raise RuntimeError(
                "isaac sim world is not configured - configure a "
                "erh:isaac-sim:world component and depend on it"
            )
        return self.cfg

    def _remote(self) -> SimClient:
        self._require_configured()
        client = self._client
        if client is None:
            raise RuntimeError("the isaac sim world is in mock mode")
        return client

    # ------------------------------------------------------------------
    # world controls (used by the world component's DoCommand)
    # ------------------------------------------------------------------

    def play(self) -> None:
        if not self._require_configured().mock:
            self._remote().call("world.play")

    def pause(self) -> None:
        if not self._require_configured().mock:
            self._remote().call("world.pause")

    def reset(self) -> None:
        if not self._require_configured().mock:
            self._remote().call("world.reset")

    def status(self) -> Dict[str, Any]:
        cfg = self.cfg
        out: Dict[str, Any] = {"configured": cfg is not None, "mock": self.mock}
        if cfg is None or cfg.mock:
            return out
        out["address"] = cfg.address
        try:
            out.update(self._remote().call("world.status"))
            out["connected"] = True
        except (ConnectionError, TimeoutError) as e:
            out["connected"] = False
            out["error"] = str(e)
        return out

    def add_usd_reference(
        self,
        usd_path: str,
        prim_path: str,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if self._require_configured().mock:
            return
        self._remote().call(
            "world.add_usd",
            {"usd_path": usd_path, "prim_path": prim_path, "position": list(position)},
            timeout=ATTACH_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # component factories
    # ------------------------------------------------------------------

    def _mock_handle(self, name: str, factory: Callable[[], Any]) -> Any:
        with self._lock:
            handle = self._mock_handles.get(name)
            if handle is None:
                handle = self._mock_handles[name] = factory()
            return handle

    def create_arm(self, name: str, attrs: Dict[str, Any]) -> "ArmHandle":
        if self._require_configured().mock:
            return self._mock_handle(name, lambda: MockArmHandle(name, attrs))
        return RemoteArmHandle(self, name, attrs)

    def create_camera(self, name: str, attrs: Dict[str, Any]) -> "CameraHandle":
        if self._require_configured().mock:
            return self._mock_handle(name, lambda: MockCameraHandle(name, attrs))
        return RemoteCameraHandle(self, name, attrs)

    def create_base(self, name: str, attrs: Dict[str, Any]) -> "BaseHandle":
        if self._require_configured().mock:
            return self._mock_handle(name, lambda: MockBaseHandle(name, attrs))
        return RemoteBaseHandle(self, name, attrs)


# ======================================================================
# Handles - the interface component models talk to. All public methods are
# safe to call from any thread.
# ======================================================================


class ArmHandle:
    def get_joint_positions(self) -> List[float]:  # radians
        raise NotImplementedError

    def set_joint_targets(self, positions: List[float]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def get_end_pose(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """((x,y,z) meters, (w,x,y,z) quaternion) of the end effector."""
        raise NotImplementedError


class CameraHandle:
    def get_image(self, mime_type: str = "") -> Tuple[bytes, str]:
        """An encoded frame and its mime type (JPEG unless PNG is asked for)."""
        raise NotImplementedError


class BaseHandle:
    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError


# how far (m) an existing prim may sit from its configured position before
# we say so
_POSE_TOLERANCE_M = 1e-3


def pose_mismatch(configured: Dict[str, Any], info: Dict[str, Any]) -> Optional[str]:
    """Why an existing prim's pose disagrees with the config, if it does.
    Prims that already exist are attached where they are, not moved."""
    problems = []
    wanted, actual = configured.get("position"), info.get("position")
    if wanted is not None and actual is not None:
        if (
            max(abs(float(w) - float(a)) for w, a in zip(wanted, actual))
            > _POSE_TOLERANCE_M
        ):
            problems.append(
                f"position {_fmt(actual)} m, not the configured {_fmt(wanted)} m"
            )
    wanted, actual = configured.get("orientation_wxyz"), info.get("orientation_wxyz")
    if wanted is not None and actual is not None:
        # q and -q are the same rotation
        dot = abs(sum(float(w) * float(a) for w, a in zip(wanted, actual)))
        if 1.0 - dot > 1e-4:
            problems.append(
                f"orientation (wxyz) {_fmt(actual)}, not the configured {_fmt(wanted)}"
            )
    return "; ".join(problems) or None


def _fmt(values: Any) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"


class _RemoteHandle:
    """A component's counterpart in the extension. Remembers how it attached
    so it can re-attach transparently when the extension no longer knows it
    (Isaac Sim restarted, or a different stage was opened)."""

    KIND = ""

    def __init__(self, sim: SimManager, name: str, attrs: Dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self._attrs = attrs
        self._attach()

    def _attach_params(self) -> Dict[str, Any]:
        return self._attrs

    def _configured_pose(self) -> Dict[str, Any]:
        return self._attrs

    def _attach(self) -> None:
        info = self._sim._remote().call(
            f"{self.KIND}.attach",
            {"name": self.name, "attrs": self._attach_params()},
            timeout=ATTACH_TIMEOUT,
        )
        self._attached(info)

    def _attached(self, info: Dict[str, Any]) -> None:
        if info.get("spawned"):
            LOGGER.info(
                "%s %r: spawned at %s", self.KIND, self.name, info.get("prim_path")
            )
            return
        problem = pose_mismatch(self._configured_pose(), info)
        if problem:
            LOGGER.warning(
                "%s %r: %s already exists at %s; it was left where it is - move "
                "the prim or update the frame config, or viam's frame system "
                "will disagree with the sim",
                self.KIND,
                self.name,
                info.get("prim_path"),
                problem,
            )

    def _call_with_payload(
        self, method: str, timeout: float = CALL_TIMEOUT, **params: Any
    ) -> Tuple[Any, bytes]:
        params["name"] = self.name
        client = self._sim._remote()
        try:
            return client.call_with_payload(f"{self.KIND}.{method}", params, timeout)
        except NotAttachedError:
            LOGGER.info("re-attaching %s %r to isaac sim", self.KIND, self.name)
            self._attach()
            return client.call_with_payload(f"{self.KIND}.{method}", params, timeout)

    def _call(self, method: str, timeout: float = CALL_TIMEOUT, **params: Any) -> Any:
        return self._call_with_payload(method, timeout, **params)[0]


class RemoteArmHandle(_RemoteHandle, ArmHandle):
    KIND = "arm"

    def get_joint_positions(self) -> List[float]:
        return [float(v) for v in self._call("get_joint_positions")]

    def set_joint_targets(self, positions: List[float]) -> None:
        self._call("set_joint_targets", positions=[float(v) for v in positions])

    def stop(self) -> None:
        self._call("stop")

    def get_end_pose(self):
        pose = self._call("get_end_pose")
        return tuple(pose["position"]), tuple(pose["orientation_wxyz"])


def camera_placement(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Where a newly created camera goes, resolved into explicit poses for the
    extension (see the attribute docs in models/camera.py)."""
    spawn: Dict[str, Any] = {}
    if attrs.get("position") is not None:
        spawn["position"] = list(to_vec3(attrs["position"]))
    if attrs.get("orientation_rpy_deg") is not None:
        spawn["orientation_wxyz"] = list(
            quat_from_euler_deg(*to_vec3(attrs["orientation_rpy_deg"]))
        )
    out: Dict[str, Any] = {"spawn_pose": spawn}

    if attrs.get("parent_prim"):
        # camera rides a (possibly moving) link; pose is local to it.
        # default: small standoff along the link, looking out the +Z
        # (tool) axis - 180deg about X flips the usd camera's -Z forward.
        local_pos = to_vec3(attrs.get("local_position"), default=(0.0, 0.0, 0.05))
        r, p, y = to_vec3(
            attrs.get("local_orientation_rpy_deg"), default=(180.0, 0.0, 0.0)
        )
        out["local_pose"] = {
            "position": list(local_pos),
            "orientation_wxyz": list(quat_from_euler_deg(r, p, y)),
        }
    elif attrs.get("target") is not None:
        # aim at a target point (world axes: +X forward, +Z up)
        position = to_vec3(attrs.get("position"), default=(3.0, 3.0, 2.5))
        out["world_pose"] = {
            "position": list(position),
            "orientation_wxyz": list(look_at_quat(position, to_vec3(attrs["target"]))),
        }
    elif attrs.get("orientation_wxyz") is not None:
        out["world_pose"] = {
            "position": list(to_vec3(attrs.get("position"))),
            "orientation_wxyz": [float(v) for v in attrs["orientation_wxyz"]],
        }
    return out


class RemoteCameraHandle(_RemoteHandle, CameraHandle):
    KIND = "camera"

    def _attach_params(self) -> Dict[str, Any]:
        return {**self._attrs, **camera_placement(self._attrs)}

    def _configured_pose(self) -> Dict[str, Any]:
        if self._attrs.get("parent_prim"):
            return {}  # posed relative to its parent link
        placement = camera_placement(self._attrs)
        pose = placement.get("world_pose") or placement["spawn_pose"]
        return {"position": pose.get("position")}

    def get_image(self, mime_type: str = "") -> Tuple[bytes, str]:
        import numpy as np

        info, payload = self._call_with_payload("get_image", mime_type=mime_type)
        if info.get("mime_type") == MIME_RAW_RGB:
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(
                int(info["height"]), int(info["width"]), 3
            )
            return encode_rgb(rgb, mime_type)
        return payload, info["mime_type"]


class RemoteBaseHandle(_RemoteHandle, BaseHandle):
    KIND = "base"

    def _attached(self, info: Dict[str, Any]) -> None:
        super()._attached(info)
        self.wheel_radius = float(info.get("wheel_radius", 0.05))
        self.wheel_base = float(info.get("wheel_base", 0.3))

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        self._call("set_velocity", linear=float(linear_mps), angular=float(angular_rps))

    def stop(self) -> None:
        self._call("stop")

    def is_moving(self) -> bool:
        return bool(self._call("is_moving"))


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

    def stop(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._start = self._positions_at(now)
            self._target = list(self._start)
            self._t0 = now

    def get_end_pose(self):
        # a fixed, deterministic pose for testing
        return ((0.3, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0))


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

    def get_image(self, mime_type: str = "") -> Tuple[bytes, str]:
        return encode_rgb(self.get_rgb(), mime_type)


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
