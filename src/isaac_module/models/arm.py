"""viam-labs:isaac-sim:arm - a simulated arm.

Attributes:
  world (string, required)         - name of the viam-labs:isaac-sim:world component
  asset (string)                   - known robot, e.g. "lite6", "ur20", "ur10", "franka"
  usd_path (string)                - explicit USD to spawn instead of a known asset;
                                     a .urdf is imported with Isaac's URDF importer
  urdf_fix_base (bool)             - for a .urdf usd_path, fix its root link to the
                                     world on import (default false)
  urdf_drive (object)              - for a .urdf usd_path, joint drive gains:
                                     {"natural_frequency": rad/s (default 100),
                                      "damping_ratio": (default 1.0, critical)}
  prim_path (string)               - where to place it (default /World/<name>), or
                                     an existing articulation in the stage
  position ([x,y,z] meters)        - spawn position
  end_effector_prim (string)       - prim path whose world pose GetEndPosition reports;
                                     known assets provide a default when available
  arm_model (string)               - optional episode manifest label for a custom USD
  fixed_base_link (string)         - link under prim_path to fix to the world at the
                                     spawn pose, for floating-root USDs whose base
                                     moves through virtual joints (RidgebackFranka: "world")
  joint_names ([string])           - articulation DOFs this arm drives, in order;
                                     overrides the known asset's joints (default:
                                     all DOFs). Lets several arms share one
                                     prim_path, each owning a joint subset.
  move_timeout_sec (float)         - max time to wait for a move (default 30)
  max_joint_deg_s (float)          - per-joint speed limit (default 180): moves
                                     follow a trapezoidal speed profile with the
                                     fastest joint at this speed, and it is the
                                     velocity limit reported to trajectory
                                     parameterizers
  max_joint_deg_s2 (float)         - per-joint acceleration limit of that
                                     profile (default 720), also reported for
                                     trajectory parameterization
  is_moving_velocity_threshold_rad_s (float) - velocity above which IsMoving
                                     reports true (default 0.01)
  robot_control_freq_hz (float)    - deprecated and ignored: moves are stepped
                                     by the simulator's physics callback
  path_tolerance_delta_deg (float) - final-waypoint convergence tolerance
                                     (default 0.5)
  path_max_step_deg (float)        - deprecated and ignored, like
                                     robot_control_freq_hz
  kinematics_url (string)          - where to fetch the kinematics file served
                                     by GetKinematics (.json = SVA, .urdf =
                                     URDF; file:// URLs work). Known assets with
                                     official Viam kinematics fetch them automatically.
                                     Without one, an arm whose base is a link (a .urdf
                                     usd_path, or joint_names on a shared prim_path)
                                     derives URDF kinematics from the live stage: its
                                     joints from the first joint's parent link to
                                     end_effector_prim, each link's collision prims'
                                     convex hulls (else its visible prims') as its
                                     planning geometry
  planning_geometry (string)       - derived kinematics: "shapes" (default), a box
                                     fitted to each link's hulls, or "mesh", the
                                     hulls themselves (served as PLY with
                                     GetKinematics)
  visual_models (bool)             - derived kinematics: serve each link's visible
                                     prims as a GLB through Get3DModels, for the 3D
                                     viewer (default false)
  initial_joint_positions_deg ([float]) - joint positions the arm spawns at
                                     and returns to on world reset, in the
                                     asset's joint order (default all zero;
                                     known assets only)
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
from ..stage_geometry import build_urdf, visual_glbs
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component

_DEFAULT_TOLERANCE_DEG = 0.5
_DEFAULT_CONTROL_FREQ_HZ = 50.0
_MAX_CONTROL_FREQ_HZ = 1000.0
_DEFAULT_MAX_STEP_DEG = 2.0
# Used when the config sets no limit, so a simulated arm moves like a real one
# instead of snapping to its target: a typical collaborative arm's joint speed,
# reached in a quarter of a second.
_DEFAULT_MAX_JOINT_DEG_S = 180.0
_DEFAULT_MAX_JOINT_DEG_S2 = 720.0
_MAX_MAX_STEP_DEG = 30.0


def _move_flags(kwargs: Mapping[str, Any]) -> Tuple[bool, bool]:
    """(waitAtEnd, interpolate) from a move's `extra`, both default true.

    The keys follow viam-ufactory-xarm, which the RDK motion service's teleop
    pipeline targets: it sends {"waitAtEnd": false, "interpolate": false} so
    each batch lands as one raw setpoint and the call returns at once."""
    extra = kwargs.get("extra") or {}
    return bool(extra.get("waitAtEnd", True)), bool(extra.get("interpolate", True))


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


def _initial_joint_positions_deg(
    attrs: Mapping[str, Any], component_name: str
) -> List[float] | None:
    raw = attrs.get("initial_joint_positions_deg")
    if raw is None:
        return None
    if (
        not isinstance(raw, (list, tuple))
        or not raw
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in raw
        )
    ):
        raise ValueError(
            f"{component_name}: initial_joint_positions_deg must be a non-empty "
            "list of finite numbers"
        )
    return [float(value) for value in raw]


def _joint_names(
    attrs: Mapping[str, Any], component_name: str
) -> list[str] | None:
    raw = attrs.get("joint_names")
    if raw is None:
        return None
    if (
        not isinstance(raw, (list, tuple))
        or not raw
        or not all(isinstance(name, str) and name for name in raw)
    ):
        raise ValueError(
            f"{component_name}: joint_names must be a non-empty list of strings"
        )
    if len(set(raw)) != len(raw):
        raise ValueError(f"{component_name}: joint_names must be unique")
    return list(raw)


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


def _scales(handle: Any, count: int) -> list[float]:
    """Multipliers from native joint units (rad | m) to Viam's (deg | mm)."""
    flags = getattr(handle, "prismatic", ()) or (False,) * count
    return [1000.0 if flag else 180.0 / math.pi for flag in flags]


class IsaacArm(Arm, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "arm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[ArmHandle] = None
        self._attrs: Dict[str, Any] = {}
        self._move_timeout = 30.0
        self._tolerance_rad = math.radians(_DEFAULT_TOLERANCE_DEG)
        self._kinematics: Optional[Tuple[Any, ...]] = None
        self._chain: Optional[Dict[str, Any]] = None  # derived kinematics' chain, for Get3DModels
        self._models: Optional[Dict[str, bytes]] = None
        self._generation = 0  # bumped by reconfigure: a load started before it is stale
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
        _initial_joint_positions_deg(attrs, config.name)
        _joint_names(attrs, config.name)
        if attrs.get("planning_geometry", "shapes") not in ("shapes", "mesh"):
            raise ValueError(f'{config.name}: planning_geometry must be "shapes" or "mesh"')
        if not isinstance(attrs.get("visual_models", False), bool):
            raise ValueError(f"{config.name}: visual_models must be true or false")
        if attrs.get("kinematics_url") and (
            attrs.get("planning_geometry") == "mesh" or attrs.get("visual_models")
        ):
            raise ValueError(
                f'{config.name}: planning_geometry "mesh" and visual_models are only for '
                "derived kinematics (no kinematics_url)"
            )
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
        step_deg = float(attrs.get("path_max_step_deg", _DEFAULT_MAX_STEP_DEG))
        if step_deg <= 0 or step_deg > _MAX_MAX_STEP_DEG:
            raise ValueError(
                f"path_max_step_deg must be in (0, {_MAX_MAX_STEP_DEG}]; got {step_deg}"
            )
        self._attrs = attrs
        attrs["is_moving_velocity_threshold_rad_s"] = (
            _is_moving_velocity_threshold(attrs, config.name)
        )
        attrs["max_joint_deg_s"] = (
            _max_joint_deg_s(attrs, config.name) or _DEFAULT_MAX_JOINT_DEG_S
        )
        attrs["max_joint_deg_s2"] = (
            _max_joint_deg_s2(attrs, config.name) or _DEFAULT_MAX_JOINT_DEG_S2
        )
        initial_joint_positions_deg = _initial_joint_positions_deg(attrs, config.name)
        if initial_joint_positions_deg is not None:
            attrs["initial_joint_positions_deg"] = initial_joint_positions_deg
        self._kinematics, self._chain, self._models = None, None, None
        self._generation += 1
        self._handle = SimManager.get().create_arm(self.name, attrs)

    async def close(self) -> None:
        """viam-server removed or renamed this arm: stop it, drop its cached
        handle and its claim on a shared articulation's USD."""
        handle, self._handle = self._handle, None
        try:
            if handle is not None:
                # else its trajectory callback keeps writing the shared joints
                await asyncio.to_thread(handle.stop)
        finally:
            SimManager.get().remove_arm(self.name)

    def _h(self) -> ArmHandle:
        if self._handle is None:
            raise RuntimeError(f"arm {self.name} is not attached to the sim")
        return self._handle

    def policy_metadata(self) -> Mapping[str, Any]:
        """Return the simulator embodiment identity used by policy validation."""
        model = str(self._attrs.get("asset", self._attrs.get("arm_model", "")))
        return {
            "simulation": True,
            "arm_model": model,
            # what the handle drives: a joint_names change needs a restart
            "joint_names": list(self._h().joint_names()),
        }

    async def get_end_position(self, **kwargs) -> Pose:
        (x, y, z), quat = await asyncio.to_thread(self._h().get_end_pose_in_base)
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
        handle = self._h()
        targets = [
            v / s for v, s in zip(positions.values, _scales(handle, len(positions.values)))
        ]
        wait_at_end, _ = _move_flags(kwargs)
        if not wait_at_end:
            await asyncio.to_thread(handle.set_joint_targets, targets)
            return
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
        """Follow the waypoints under the arm's speed and acceleration limits
        and stop at the last one; the simulator steps the motion, so it takes
        the same simulated time whatever rate the sim runs at.

        `extra` {"waitAtEnd": false} returns without waiting for the arm to
        arrive; a newer move or Stop replaces the motion. {"interpolate":
        false} goes straight to the last waypoint. Both together (the RDK
        teleop pipeline) hand it to the sim as a servo setpoint: the call
        returns at once and the arm interpolates between successive setpoints."""
        handle = self._h()
        raw = [
            [v / s for v, s in zip(wp.values, _scales(handle, len(wp.values)))]
            for wp in positions
        ]
        if not raw:
            return
        wait_at_end, interpolate = _move_flags(kwargs)

        if not interpolate and not wait_at_end:
            await asyncio.to_thread(handle.servo_target, raw[-1])
            return
        await asyncio.to_thread(handle.follow_path, raw if interpolate else raw[-1:])
        if not wait_at_end:
            return
        self._moving_count += 1
        try:
            await self._await_final(handle, raw)
        finally:
            self._moving_count -= 1

    async def _await_final(
        self, handle: ArmHandle, raw: Sequence[Sequence[float]]
    ) -> None:
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
                            float(value) / scale
                            for value, scale in zip(
                                point.positions, _scales(handle, len(point.positions))
                            )
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
        handle = self._h()
        native = await asyncio.to_thread(handle.get_joint_positions)
        return JointPositions(
            values=[v * s for v, s in zip(native, _scales(handle, len(native)))]
        )

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        """True while a MoveTo* RPC is in flight or a waitAtEnd=false move
        is still playing."""
        if self._moving_count > 0:
            return True
        return self._handle is not None and self._handle.path_active()

    def _kinematics_url(self) -> Optional[str]:
        url = self._attrs.get("kinematics_url")
        if url:
            return str(url)
        asset = self._attrs.get("asset")
        if asset and asset in KNOWN_ASSETS:
            return KNOWN_ASSETS[asset].get("kinematics")
        return None

    def _derive_kinematics(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        chain = self._h().stage_chain(self.name, bool(self._attrs.get("visual_models")))
        urdf, files = build_urdf(chain, self._attrs.get("planning_geometry") == "mesh")
        fmt = KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF
        if not files:
            return (fmt, urdf), chain
        # RDK reads a URDF's mesh collisions from these, keyed by filename
        meshes = {path: Mesh(content_type=kind, mesh=data) for path, (kind, data) in files.items()}
        return (fmt, urdf, meshes), chain

    def _load_kinematics(self) -> Tuple[Tuple[Any, ...], Optional[Dict[str, Any]]]:
        """The kinematics GetKinematics serves, and a derived arm's chain."""
        url = self._kinematics_url()
        if not url:
            return self._derive_kinematics()
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
                return (fmt, f.read()), None

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
        return (fmt, data), None

    async def get_kinematics(self, **kwargs) -> Any:
        if self._kinematics is not None:
            return self._kinematics
        generation = self._generation
        kinematics, chain = await asyncio.to_thread(self._load_kinematics)
        if generation == self._generation:  # else reconfigured meanwhile: don't keep it
            self._kinematics, self._chain = kinematics, chain
        return kinematics

    async def get_3d_models(self, **kwargs) -> Mapping[str, Mesh]:
        if not self._attrs.get("visual_models"):
            return {}
        await self.get_kinematics()
        models = self._models
        if models is None:
            if self._chain is None:
                return {}
            generation, meshes = self._generation, self._attrs.get("planning_geometry") == "mesh"
            try:
                models = await asyncio.to_thread(visual_glbs, self._chain, meshes)
            except ImportError as exc:  # trimesh comes with isaacsim, not this module
                self.logger.warning("%s: visual_models needs trimesh (%s); serving none", self.name, exc)
                return {}
            if generation == self._generation:
                self._models = models
        return {
            link: Mesh(content_type="model/gltf-binary", mesh=glb)
            for link, glb in models.items()
        }

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
            dof = len(self._h().joint_names())
            if not isinstance(raw, (list, tuple)) or len(raw) != dof:
                raise ValueError(f"set_joint_targets_radians requires {dof} values")
            targets = [float(value) for value in cast(Any, raw)]
            if not all(math.isfinite(value) for value in targets):
                raise ValueError("set_joint_targets_radians values must be finite")
            await asyncio.to_thread(self._h().set_joint_targets, targets)
            return {"accepted": True}
        raise ValueError(f"unknown command: {command}")
