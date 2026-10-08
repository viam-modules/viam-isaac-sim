"""viam-labs:isaac-sim:arm - a simulated arm.

Attributes:
  world (string, required)         - name of the viam-labs:isaac-sim:world component
  asset (string)                   - known robot, e.g. "ur20", "ur10", "franka"
  usd_path (string)                - explicit USD to spawn instead of a known asset
  prim_path (string)               - where to place it (default /World/<name>), or
                                     an existing articulation in the stage
  position ([x,y,z] meters)        - spawn position
  end_effector_prim (string)       - prim path whose world pose is reported by
                                     GetEndPosition
  move_timeout_sec (float)         - max time to wait for a move (default 30)
  robot_control_freq_hz (float)    - rate at which MoveThroughJointPositions
                                     streams waypoint targets (default 50Hz,
                                     range (0, 1000])
  path_tolerance_delta_deg (float) - final-waypoint convergence tolerance
                                     (default 0.5)
  path_max_step_deg (float)        - max per-joint delta between two pushed
                                     targets; sparse waypoint pairs are
                                     linearly interpolated so drives can
                                     actually track them (default 2.0,
                                     range (0, 30])
  kinematics_url (string)          - where to fetch the kinematics file served
                                     by GetKinematics (.json = SVA, .urdf =
                                     URDF; file:// URLs work). Known assets
                                     with official viam kinematics
                                     (ur3e/ur5e/ur20) fetch them automatically.
"""

import asyncio
import hashlib
import math
import os
import tempfile
import time
import urllib.request
from typing import (
    Any,
    AsyncIterator,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from typing_extensions import Self
from viam.components.arm import Arm, JointPositions, KinematicsFileFormat, Mesh, Pose
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..sim_manager import KNOWN_ASSETS, ArmHandle, SimManager
from ..spatial import quat_to_ov
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component

_DEFAULT_TOLERANCE_DEG = 0.5
_DEFAULT_CONTROL_FREQ_HZ = 50.0
_MAX_CONTROL_FREQ_HZ = 1000.0
_DEFAULT_MAX_STEP_DEG = 2.0
_MAX_MAX_STEP_DEG = 30.0


def _densify(
    start: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    max_step_rad: float,
) -> List[List[float]]:
    """Linearly interpolate targets between consecutive waypoints so no single
    step moves any joint by more than `max_step_rad`. First segment runs from
    `start` (the arm's current joint positions) to `waypoints[0]`; the last
    emitted target equals `waypoints[-1]` exactly."""
    out: List[List[float]] = []
    prev: List[float] = list(start)
    for wp in waypoints:
        target = list(wp)
        max_delta = max((abs(a - b) for a, b in zip(prev, target)), default=0.0)
        n = max(1, math.ceil(max_delta / max_step_rad))
        for k in range(1, n + 1):
            t = k / n
            out.append([a + t * (b - a) for a, b in zip(prev, target)])
        prev = target
    return out


class IsaacArm(Arm, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "arm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[ArmHandle] = None
        self._attrs: Dict[str, Any] = {}
        self._move_timeout = 30.0
        self._tolerance_rad = math.radians(_DEFAULT_TOLERANCE_DEG)
        self._control_period = 1.0 / _DEFAULT_CONTROL_FREQ_HZ
        self._max_step_rad = math.radians(_DEFAULT_MAX_STEP_DEG)
        self._kinematics: Optional[Tuple[KinematicsFileFormat.ValueType, bytes]] = None
        self._moving_count = 0

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        arm = cls(config.name)
        arm.reconfigure(config, dependencies)
        return arm

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        return validate_sim_component(config)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._move_timeout = float(attrs.get("move_timeout_sec", 30.0))
        self._tolerance_rad = math.radians(
            float(attrs.get("path_tolerance_delta_deg", _DEFAULT_TOLERANCE_DEG))
        )
        freq = float(attrs.get("robot_control_freq_hz", _DEFAULT_CONTROL_FREQ_HZ))
        if freq <= 0 or freq > _MAX_CONTROL_FREQ_HZ:
            raise ValueError(
                f"robot_control_freq_hz must be in (0, {_MAX_CONTROL_FREQ_HZ}]; got {freq}"
            )
        self._control_period = 1.0 / freq
        step_deg = float(attrs.get("path_max_step_deg", _DEFAULT_MAX_STEP_DEG))
        if step_deg <= 0 or step_deg > _MAX_MAX_STEP_DEG:
            raise ValueError(
                f"path_max_step_deg must be in (0, {_MAX_MAX_STEP_DEG}]; got {step_deg}"
            )
        self._max_step_rad = math.radians(step_deg)
        self._attrs = attrs
        self._handle = SimManager.get().create_arm(self.name, attrs)

    def _h(self) -> ArmHandle:
        if self._handle is None:
            raise RuntimeError(f"arm {self.name} is not attached to the sim")
        return self._handle

    async def get_end_position(self, **kwargs) -> Pose:
        (x, y, z), quat = await asyncio.to_thread(self._h().get_end_pose)
        ox, oy, oz, theta = quat_to_ov(quat)
        return Pose(
            x=x * 1000.0,
            y=y * 1000.0,
            z=z * 1000.0,
            o_x=ox,
            o_y=oy,
            o_z=oz,
            theta=math.degrees(theta),
        )

    async def move_to_position(self, pose: Pose, **kwargs) -> None:
        raise NotImplementedError(
            "IK and motion planning are Viam's job, not the module's: use the "
            "motion service (needs GetKinematics, on the roadmap) or "
            "move_to_joint_positions"
        )

    async def move_to_joint_positions(
        self, positions: JointPositions, **kwargs
    ) -> None:
        targets = [math.radians(v) for v in positions.values]
        handle = self._h()
        self._moving_count += 1
        try:
            await asyncio.to_thread(handle.set_joint_targets, targets)

            deadline = time.monotonic() + self._move_timeout
            while time.monotonic() < deadline:
                current = await asyncio.to_thread(handle.get_joint_positions)
                if len(current) >= len(targets) and all(
                    abs(c - t) <= self._tolerance_rad for c, t in zip(current, targets)
                ):
                    return
                await asyncio.sleep(0.05)
            raise TimeoutError(
                f"arm {self.name} did not reach target within {self._move_timeout}s"
            )
        finally:
            self._moving_count -= 1

    async def move_through_joint_positions(
        self, positions: Sequence[JointPositions], *args, **kwargs
    ) -> None:
        """Stream waypoint targets at `robot_control_freq_hz`. Sparse waypoint
        pairs are linearly interpolated so no single pushed target moves any
        joint by more than `path_max_step_deg`; otherwise drives would cut
        corners past intermediate waypoints they cannot reach in one tick."""
        handle = self._h()
        raw = [[math.radians(v) for v in wp.values] for wp in positions]
        if not raw:
            return

        self._moving_count += 1
        try:
            start = await asyncio.to_thread(handle.get_joint_positions)
            stream = _densify(start, raw, self._max_step_rad)

            loop = asyncio.get_running_loop()
            push_start = loop.time()
            for i, targets in enumerate(stream):
                await asyncio.to_thread(handle.set_joint_targets, targets)
                target_time = push_start + (i + 1) * self._control_period
                sleep_needed = target_time - loop.time()
                if sleep_needed > 0:
                    await asyncio.sleep(sleep_needed)

            final_targets = raw[-1]
            deadline = loop.time() + self._move_timeout
            current: List[float] = []
            while loop.time() < deadline:
                current = await asyncio.to_thread(handle.get_joint_positions)
                if len(current) >= len(final_targets) and all(
                    abs(c - t) <= self._tolerance_rad
                    for c, t in zip(current, final_targets)
                ):
                    return
                await asyncio.sleep(0.05)

            detail = ", ".join(
                f"j{j}: at {math.degrees(c):.1f} want {math.degrees(t):.1f}"
                for j, (c, t) in enumerate(zip(current, final_targets))
                if abs(c - t) > self._tolerance_rad
            )
            raise TimeoutError(
                f"arm {self.name} did not converge on final waypoint "
                f"{len(raw)}/{len(raw)} within {self._move_timeout}s "
                f"(stuck joints: {detail})"
            )
        finally:
            self._moving_count -= 1

    async def move_through_joint_positions_streamed(  # type: ignore
        self,
        batches: AsyncIterator[List[Arm.TrajectoryPoint]],
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[Arm.TrajectoryUpdate]:
        raise NotImplementedError(
            f"arm {self.name} does not support move_through_joint_positions_streamed"
        )
        # unreachable, but makes this an async generator so the RPC handler's
        # `async for` surfaces the NotImplementedError instead of a TypeError
        yield Arm.TrajectoryUpdate()

    async def get_joint_positions(self, **kwargs) -> JointPositions:
        radians = await asyncio.to_thread(self._h().get_joint_positions)
        return JointPositions(values=[math.degrees(r) for r in radians])

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        """True while a MoveTo* RPC is in flight."""
        return self._moving_count > 0

    def _kinematics_url(self) -> Optional[str]:
        url = self._attrs.get("kinematics_url")
        if url:
            return str(url)
        asset = self._attrs.get("asset")
        if asset and asset in KNOWN_ASSETS:
            return KNOWN_ASSETS[asset].get("kinematics")
        return None

    def _load_kinematics(self) -> Tuple[KinematicsFileFormat.ValueType, bytes]:
        url = self._kinematics_url()
        if not url:
            raise NotImplementedError(
                f"no kinematics file known for arm {self.name}; set the "
                '"kinematics_url" attribute (SVA .json or .urdf)'
            )
        ext = os.path.splitext(url)[1].lower()
        fmt = (
            KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF
            if ext in (".urdf", ".xml", ".xacro")
            else KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
        )

        cache_dir = os.environ.get("VIAM_MODULE_DATA") or tempfile.gettempdir()
        cache = os.path.join(
            cache_dir,
            f"kinematics-{hashlib.sha1(url.encode()).hexdigest()[:12]}{ext}",
        )
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return fmt, f.read()

        self.logger.info("fetching kinematics for %s from %s", self.name, url)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cache + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, cache)
        except OSError:
            pass  # caching is best-effort
        return fmt, data

    async def get_kinematics(self, **kwargs) -> Tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            self._kinematics = await asyncio.to_thread(self._load_kinematics)
        return self._kinematics

    async def get_3d_models(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, Mesh]:
        raise NotImplementedError(f"arm {self.name} does not support get_3d_models")

    async def set_manual_mode(
        self,
        manual_mode: bool,
        enabled_for: int = 0,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError(f"arm {self.name} does not support manual mode")

    async def get_manual_mode(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> bool:
        raise NotImplementedError(f"arm {self.name} does not support manual mode")

    async def get_properties(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Arm.Properties:
        raise NotImplementedError(f"arm {self.name} does not support get_properties")

    async def get_geometries(self, **kwargs) -> List[Geometry]:
        return []

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") == "get_joint_positions_radians":
            return {"values": await asyncio.to_thread(self._h().get_joint_positions)}
        raise ValueError(f"unknown command: {command}")
