"""viam-labs:isaac-sim:arm - a simulated arm.

Attributes:
  world (string, required)         - name of the viam-labs:isaac-sim:world component
  asset (string)                   - known robot, e.g. "lite6", "ur20", "ur10", "franka"
  usd_path (string)                - explicit USD to spawn instead of a known asset
  prim_path (string)               - where to place it (default /World/<name>), or
                                     an existing articulation in the stage
  position ([x,y,z] meters)        - spawn position
  end_effector_prim (string)       - prim path whose world pose GetEndPosition reports;
                                     known assets provide a default when available
  arm_model (string)               - optional episode manifest label for a custom USD
  move_timeout_sec (float)         - max time to wait for a move (default 30)
  max_joint_deg_s (float)          - optional positive per-joint velocity limit passed
                                     to the Isaac articulation
  max_joint_deg_s2 (float)         - optional positive per-joint acceleration limit
                                     reported for trajectory parameterization
  is_moving_velocity_threshold_rad_s (float) - velocity above which IsMoving
                                     reports true (default 0.01)
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
                                     URDF; file:// URLs work). Known assets with
                                     official Viam kinematics fetch them automatically.
"""

import asyncio
import hashlib
import math
import os
import tempfile
import urllib.request
from datetime import timedelta
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
    cast,
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
from ..sim_manager import (
    DEFAULT_ARM_IS_MOVING_VELOCITY_THRESHOLD_RAD_S,
    KNOWN_ASSETS,
    ArmHandle,
    JointTrajectoryPoint,
    SimManager,
)
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


def _max_joint_deg_s(
    attrs: Mapping[str, Any], component_name: str
) -> float | None:
    raw = attrs.get("max_joint_deg_s")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{component_name}: max_joint_deg_s must be numeric")
    limit = float(raw)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            f"{component_name}: max_joint_deg_s must be finite and positive"
        )
    return limit


def _max_joint_deg_s2(
    attrs: Mapping[str, Any], component_name: str
) -> float | None:
    raw = attrs.get("max_joint_deg_s2")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{component_name}: max_joint_deg_s2 must be numeric")
    limit = float(raw)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            f"{component_name}: max_joint_deg_s2 must be finite and positive"
        )
    return limit


def _is_moving_velocity_threshold(
    attrs: Mapping[str, Any], component_name: str
) -> float:
    raw = attrs.get(
        "is_moving_velocity_threshold_rad_s",
        DEFAULT_ARM_IS_MOVING_VELOCITY_THRESHOLD_RAD_S,
    )
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"{component_name}: is_moving_velocity_threshold_rad_s must be numeric"
        )
    threshold = float(raw)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            f"{component_name}: is_moving_velocity_threshold_rad_s must be finite "
            "and non-negative"
        )
    return threshold


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
        result = validate_sim_component(config)
        attrs = get_attrs(config)
        _is_moving_velocity_threshold(attrs, config.name)
        _max_joint_deg_s(attrs, config.name)
        _max_joint_deg_s2(attrs, config.name)
        return result

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
        attrs["is_moving_velocity_threshold_rad_s"] = (
            _is_moving_velocity_threshold(attrs, config.name)
        )
        max_joint_deg_s = _max_joint_deg_s(attrs, config.name)
        if max_joint_deg_s is not None:
            attrs["max_joint_deg_s"] = max_joint_deg_s
        max_joint_deg_s2 = _max_joint_deg_s2(attrs, config.name)
        if max_joint_deg_s2 is not None:
            attrs["max_joint_deg_s2"] = max_joint_deg_s2
        self._handle = SimManager.get().create_arm(self.name, attrs)

    def _h(self) -> ArmHandle:
        if self._handle is None:
            raise RuntimeError(f"arm {self.name} is not attached to the sim")
        return self._handle

    def policy_metadata(self) -> Mapping[str, Any]:
        """Return the simulator embodiment identity used by policy validation."""
        model = str(self._attrs.get("asset", self._attrs.get("arm_model", "")))
        meta = KNOWN_ASSETS.get(model, {})
        return {
            "simulation": True,
            "arm_model": model,
            "joint_names": list(meta.get("joint_names", ())),
        }

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

    async def _wait_for_joint_targets(
        self,
        handle: ArmHandle,
        targets: Sequence[float],
        tolerance_rad: float,
        timeout_s: float,
    ) -> bool:
        """Block until the sim thread reports the joints within tolerance.

        The convergence check runs once per simulator frame on the sim thread,
        so there is no RPC polling or wall-clock sleep between checks; the
        wall-clock timeout only bounds a stalled move. Returns False on
        timeout."""
        future = await asyncio.to_thread(
            handle.wait_for_joint_targets, targets, tolerance_rad
        )
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except TimeoutError:
            future.cancel()
            return False
        return True

    @staticmethod
    def _stuck_joint_detail(
        current: Sequence[float], targets: Sequence[float], tolerance_rad: float
    ) -> str:
        return ", ".join(
            f"j{index}: at {math.degrees(actual):.1f} want {math.degrees(target):.1f}"
            for index, (actual, target) in enumerate(zip(current, targets))
            if abs(actual - target) > tolerance_rad
        )

    async def move_to_joint_positions(
        self, positions: JointPositions, **kwargs
    ) -> None:
        targets = [math.radians(v) for v in positions.values]
        handle = self._h()
        self._moving_count += 1
        try:
            await asyncio.to_thread(handle.set_joint_targets, targets)
            if await self._wait_for_joint_targets(
                handle, targets, self._tolerance_rad, self._move_timeout
            ):
                return
            current = await asyncio.to_thread(handle.get_joint_positions)
            detail = self._stuck_joint_detail(
                current, targets, self._tolerance_rad
            )
            raise TimeoutError(
                f"arm {self.name} did not reach target within {self._move_timeout}s"
                f" (stuck joints: {detail or 'position unavailable'})"
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
            if await self._wait_for_joint_targets(
                handle, final_targets, self._tolerance_rad, self._move_timeout
            ):
                return
            current = await asyncio.to_thread(handle.get_joint_positions)
            detail = self._stuck_joint_detail(
                current, final_targets, self._tolerance_rad
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
        **kwargs,
    ) -> AsyncIterator[Arm.TrajectoryUpdate]:
        last_time = timedelta(0)
        first = True
        handle = self._h()
        async for batch in batches:
            trajectory: list[JointTrajectoryPoint] = []
            batch_start = last_time
            for point in batch:
                if first:
                    if point.time != timedelta(0):
                        raise ValueError("first trajectory point must have time zero")
                    first = False
                elif point.time <= last_time:
                    raise ValueError("trajectory point times must strictly increase")
                trajectory.append(
                    JointTrajectoryPoint(
                        time_s=(point.time - batch_start).total_seconds(),
                        positions=tuple(
                            math.radians(float(value)) for value in point.positions
                        ),
                    )
                )
                last_time = point.time

            if trajectory:
                expected = trajectory[-1].positions
                start_task = asyncio.create_task(
                    asyncio.to_thread(handle.start_trajectory, trajectory)
                )
                started = False
                try:
                    future = await asyncio.shield(start_task)
                    started = True
                    await asyncio.wrap_future(future)
                finally:
                    if not started:
                        try:
                            await start_task
                        except BaseException:
                            started = False
                        else:
                            started = True
                    if started:
                        await asyncio.to_thread(handle.stop_trajectory)
                actual = await asyncio.to_thread(handle.get_joint_positions)
                error_deg = max(
                    (
                        abs(math.degrees(want - got))
                        for want, got in zip(expected, actual)
                    ),
                    default=0.0,
                )
                self.logger.info(
                    "%s: trajectory complete expected=%s actual=%s max_error_deg=%.2f",
                    self.name,
                    [round(math.degrees(value), 2) for value in expected],
                    [round(math.degrees(value), 2) for value in actual],
                    error_deg,
                )
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

    async def get_kinematics(
        self, **kwargs
    ) -> Tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            self._kinematics = await asyncio.to_thread(self._load_kinematics)
        return self._kinematics

    async def get_3d_models(self, **kwargs) -> Mapping[str, Mesh]:
        return {}

    async def set_manual_mode(
        self, manual_mode: bool, enabled_for: int = 0, **kwargs
    ) -> None:
        raise NotImplementedError("manual mode is not supported by the simulated arm")

    async def get_manual_mode(self, **kwargs) -> bool:
        return False

    async def get_properties(self, **kwargs) -> Arm.Properties:
        return Arm.Properties(
            support_manual_mode=False,
            support_cartesian_commands=False,
        )

    async def get_geometries(self, **kwargs) -> List[Geometry]:
        return []

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if cmd == "policy_metadata":
            return self.policy_metadata()
        if cmd == "get_joint_positions_radians":
            return cast(
                Mapping[str, ValueTypes],
                {"values": await asyncio.to_thread(self._h().get_joint_positions)},
            )
        if cmd == "dynamics_metadata":
            return await asyncio.to_thread(self._h().dynamics_metadata)
        if cmd == "set_joint_targets_radians":
            raw = command.get("values")
            if not isinstance(raw, (list, tuple)) or len(raw) != 6:
                raise ValueError("set_joint_targets_radians requires six values")
            targets = [float(value) for value in cast(Any, raw)]
            if not all(math.isfinite(value) for value in targets):
                raise ValueError("set_joint_targets_radians values must be finite")
            await asyncio.to_thread(self._h().set_joint_targets, targets)
            return {"accepted": True}
        raise ValueError(f"unknown command: {command}")
