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

import hashlib
import math
import os
import queue
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from viam.logging import getLogger

LOGGER = getLogger("viam-isaac-sim")

# Assets shipped on the Isaac Sim nucleus/content server, addressable by a
# short name in component config. Paths are relative to the assets root;
# where isaac 5.0 moved an asset, the 5.0 path is listed first with the 4.x
# path as a fallback - the first candidate that exists is used.
_UR_KINEMATICS = "https://raw.githubusercontent.com/viam-modules/universal-robots/main/src/kinematics"
UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
LITE6_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
LITE6_JAW_LAYOUTS: dict[str, tuple[float, float]] = {
    "narrow": (-1.0, 1.0),
    "middle_positive_y": (1.0, 1.0),
    "middle_negative_y": (-1.0, -1.0),
    "wide": (1.0, -1.0),
}
# Reversing a bracket swaps which edge carries its vertical tip while its
# 15.6 mm base remains in the same carriage footprint.
LITE6_JAW_BASE_WIDTH_M = 0.015600001
LITE6_MIMIC_NATURAL_FREQUENCY = 1000.0
LITE6_MIMIC_DAMPING_RATIO = 1.0
GRASP_CONTACT_OFFSET_M = 0.001
GRASP_REST_OFFSET_M = 0.0
GRIPPER_CONTACT_FORCE_N = 0.15
GRIPPER_HOLD_FORCE_N = 2.0
_UFACTORY_KINEMATICS = (
    "https://raw.githubusercontent.com/viam-modules/viam-ufactory-xarm/main/arm"
)

VISUAL_MATERIAL_PRESETS: dict[str, dict[str, Any]] = {
    "wood": {
        "diffuse_color": (0.36, 0.16, 0.06),
        "roughness": 0.72,
        "metallic": 0.0,
    },
    "brushed_metal": {
        "diffuse_color": (0.48, 0.5, 0.53),
        "roughness": 0.3,
        "metallic": 1.0,
    },
}


def _visual_material_properties(name: Any) -> dict[str, Any]:
    if not isinstance(name, str) or name not in VISUAL_MATERIAL_PRESETS:
        raise ValueError(
            f"visual_material must be one of {sorted(VISUAL_MATERIAL_PRESETS)}, got {name!r}"
        )
    return VISUAL_MATERIAL_PRESETS[name]


KNOWN_ASSETS: dict[str, dict[str, Any]] = {
    "ur3e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur3e.json",
        "joint_names": UR_JOINT_NAMES,
        "end_effector_link": "wrist_3_link",
    },
    "ur5e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur5e.json",
        "joint_names": UR_JOINT_NAMES,
        "end_effector_link": "wrist_3_link",
    },
    "ur10": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur10/ur10.usd"],
        "joint_names": UR_JOINT_NAMES,
        "end_effector_link": "wrist_3_link",
    },
    "ur10e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"],
        "joint_names": UR_JOINT_NAMES,
        "end_effector_link": "wrist_3_link",
    },
    "ur16e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur16e/ur16e.usd"],
        "joint_names": UR_JOINT_NAMES,
        "end_effector_link": "wrist_3_link",
    },
    "ur20": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur20/ur20.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur20.json",
        "joint_names": UR_JOINT_NAMES,
        "end_effector_link": "wrist_3_link",
    },
    "lite6": {
        "usd": ["/Isaac/Robots/Ufactory/lite6/lite6.usd"],
        "kinematics": f"{_UFACTORY_KINEMATICS}/lite6_kinematics.json",
        "joint_names": LITE6_JOINT_NAMES,
        "end_effector_link": "gripper/uflite_gripper_link",
        "variants": {"Variant_Set": "lite_gripper"},
    },
    "franka": {
        "usd": [
            "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "/Isaac/Robots/Franka/franka.usd",
        ]
    },
    "robotiq_2f_85": {
        "kind": "gripper",
        "usd": ["/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"],
        "drive_joint": "finger_joint",
        "open_deg": 0.0,
        "closed_deg": 45.0,
        "tcp_offset_m": 0.134,
        "jaw_box_mm": (36.0, 146.0, 153.0),
        "gripper_mode": "position",
    },
    "lite6_gripper": {
        "kind": "gripper",
        "usd": ["/Isaac/Robots/Ufactory/lite6_gripper/uf_lite_gripper.usd"],
        "integrated_arm_asset": "lite6",
        "drive_joint": "finger_joint1",
        "open_limit": "upper",
        "closed_limit": "lower",
        "open_position": 0.0089,
        "closed_position": 0.0,
        "position_unit": "m",
        "holding_tolerance": 0.0005,
        "tcp_offset_m": 0.0836,
        "jaw_box_mm": (110.0, 80.0, 84.0),
        "open_width_m": 0.016,
        "minimum_grasp_closure": 0.05,
        "drive_stiffness": 300.0,
        "drive_damping": 5.0,
        "coupled_drive_joints": ("finger_joint2",),
        "gripper_mode": "binary",
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


def default_rendering_dt(physics_dt: float, capture_rate_hz: float) -> float:
    """Render once per capture sample when that is a whole number of physics
    steps, otherwise once per physics step.

    Capture samples at ``capture_rate_hz`` of simulator time and only reads
    camera buffers on those ticks, so rendering more often than that spends GPU
    time on frames nobody consumes. Isaac requires ``rendering_dt`` to be a
    positive integer multiple of ``physics_dt``."""
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("physics_dt must be finite and positive")
    if not math.isfinite(capture_rate_hz) or capture_rate_hz <= 0.0:
        raise ValueError("capture_rate_hz must be finite and positive")
    steps_per_sample = (1.0 / capture_rate_hz) / physics_dt
    rounded = round(steps_per_sample)
    if rounded >= 1 and math.isclose(steps_per_sample, rounded, abs_tol=1e-9):
        return physics_dt * rounded
    return physics_dt


def validate_rendering_dt(physics_dt: float, rendering_dt: float) -> None:
    """Raise ValueError unless rendering_dt is a positive integer multiple of
    physics_dt, which Isaac's World enforces at construction."""
    if not math.isfinite(rendering_dt) or rendering_dt <= 0.0:
        raise ValueError("rendering_dt must be finite and positive")
    ratio = rendering_dt / physics_dt
    if ratio < 1.0 - 1e-9 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise ValueError(
            f"rendering_dt {rendering_dt:g} must be a positive integer multiple "
            f"of physics_dt {physics_dt:g}"
        )


@dataclass
class SimConfig:
    mock: bool = False
    headless: bool = True
    livestream: bool = True
    usd_stage: str | None = None
    dome_light: bool = False
    physics_dt: float = 1.0 / 60.0
    # None renders once per capture sample (see default_rendering_dt).
    rendering_dt: float | None = None
    capture_rate_hz: float = 15.0
    # Sample episodes after each rendered frame instead of from a physics
    # callback, with kit told to finish each render before returning. A
    # physics callback runs before that update's render, so its camera
    # reads return an older frame than its joint state.
    capture_after_render: bool = False
    boot_timeout: float = 1200.0
    # IP the livestream advertises to clients; auto-detected if empty
    livestream_public_ip: str = ""
    # resolution kit renders (and therefore streams) at
    livestream_width: int = 1280
    livestream_height: int = 720
    # props to spawn into the scene at boot; each entry:
    #   {"type": "cube"|"usd", "name": ..., "position": [x,y,z] (m),
    #    "size": edge_m, "scale": [sx,sy,sz], "color": [r,g,b] 0-1,
    #    "fixed": bool, "usd_path": ..., "static_friction": float,
    #    "dynamic_friction": float, "visual_material": "wood"|"brushed_metal"}
    props: list[dict[str, Any]] = field(default_factory=list)
    # kit console verbosity (verbose/info/warning/error). Kit prints thousands
    # of lines at info, and viam-server records the module's stderr as
    # error-level logs, so default to warning.
    kit_log_level: str = "warning"
    # add_default_ground_plane loads the Grid env USD, which bundles floor
    # collider AND the lighting rig; when false, we add fallback lights
    isaac_default_environment: bool = True

    def resolved_rendering_dt(self) -> float:
        """The render period Isaac's World is built with."""
        if self.rendering_dt is None:
            return default_rendering_dt(self.physics_dt, self.capture_rate_hz)
        validate_rendering_dt(self.physics_dt, self.rendering_dt)
        return self.rendering_dt


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    try:
        x, y, z = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly three numbers") from exc
    return float(x), float(y), float(z)


def _quaternion(value: Any, label: str) -> tuple[float, float, float, float]:
    try:
        w, x, y, z = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly four numbers") from exc
    return float(w), float(x), float(y), float(z)


def _dof_limit_pair(articulation: Any, index: int) -> tuple[float, float]:
    """Read one DOF's limits through SingleArticulation's public API."""
    properties = articulation.dof_properties
    return float(properties["lower"][index]), float(properties["upper"][index])


def _get_or_add_xform_op(
    xformable: Any, op_type: Any, add_op: Callable[[], Any]
) -> Any:
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return add_op()


def _reset_articulation_at_pose(
    world: Any,
    articulation: Any,
    position: Sequence[float],
    orientation: Sequence[float] | None,
) -> None:
    pose: dict[str, Any] = {"position": list(position)}
    if orientation is not None:
        pose["orientation"] = list(orientation)
    articulation.set_default_state(**pose)
    world.reset()
    articulation.set_world_pose(**pose)


def _parented_camera_pose(
    attrs: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], str]:
    from .spatial import quat_from_euler_deg

    local_position = _vector3(
        attrs.get("local_position", (0.0, 0.0, 0.05)), "local_position"
    )
    if attrs.get("local_orientation_wxyz") is not None:
        return (
            local_position,
            _quaternion(attrs["local_orientation_wxyz"], "local_orientation_wxyz"),
            str(attrs.get("local_camera_axes", "ros")),
        )
    roll, pitch, yaw = _vector3(
        attrs.get("local_orientation_rpy_deg", (180.0, 0.0, 0.0)),
        "local_orientation_rpy_deg",
    )
    return local_position, quat_from_euler_deg(roll, pitch, yaw), "usd"


def _values_close(actual: Sequence[float], expected: Sequence[float]) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-5)
        for left, right in zip(actual, expected)
    )


def _quaternions_close(actual: Sequence[float], expected: Sequence[float]) -> bool:
    return _values_close(actual, expected) or _values_close(
        actual, tuple(-float(value) for value in expected)
    )


def _apply_camera_configuration(camera: Any, attrs: dict[str, Any]) -> None:
    from .spatial import look_at_quat, quat_from_euler_deg

    parent = attrs.get("parent_prim")
    expected_position: Sequence[float] | None = None
    expected_orientation: Sequence[float] | None = None
    camera_axes = "world"
    if parent:
        expected_position, expected_orientation, camera_axes = _parented_camera_pose(
            attrs
        )
        camera.set_local_pose(
            list(expected_position),
            list(expected_orientation),
            camera_axes=camera_axes,
        )
        actual_position, actual_orientation = camera.get_local_pose(
            camera_axes=camera_axes
        )
    else:
        if attrs.get("target") is not None:
            expected_position = _vector3(
                attrs.get("position", (3.0, 3.0, 2.5)), "position"
            )
            expected_orientation = look_at_quat(
                expected_position, _vector3(attrs["target"], "target")
            )
        elif attrs.get("orientation_wxyz") is not None:
            expected_position = _vector3(
                attrs.get("position", (0.0, 0.0, 0.0)), "position"
            )
            expected_orientation = _quaternion(
                attrs["orientation_wxyz"], "orientation_wxyz"
            )
        elif attrs.get("orientation_rpy_deg") is not None:
            expected_position = _vector3(
                attrs.get("position", (0.0, 0.0, 0.0)), "position"
            )
            roll, pitch, yaw = _vector3(
                attrs["orientation_rpy_deg"], "orientation_rpy_deg"
            )
            expected_orientation = quat_from_euler_deg(roll, pitch, yaw)
        elif attrs.get("position") is not None:
            expected_position = _vector3(attrs["position"], "position")
        if expected_position is not None or expected_orientation is not None:
            camera.set_world_pose(
                list(expected_position) if expected_position is not None else None,
                list(expected_orientation)
                if expected_orientation is not None
                else None,
                camera_axes=camera_axes,
            )
        actual_position, actual_orientation = camera.get_world_pose(
            camera_axes=camera_axes
        )

    if expected_position is not None and not _values_close(
        actual_position, expected_position
    ):
        raise RuntimeError(
            f"camera position update did not apply: {tuple(actual_position)!r}"
        )
    if expected_orientation is not None and not _quaternions_close(
        actual_orientation, expected_orientation
    ):
        raise RuntimeError(
            f"camera orientation update did not apply: {tuple(actual_orientation)!r}"
        )

    near_clip_m = float(attrs.get("near_clip_m", 0.01))
    if near_clip_m <= 0.0:
        raise ValueError("camera near_clip_m must be positive")
    camera.set_clipping_range(near_distance=near_clip_m)
    actual_near, _ = camera.get_clipping_range()
    if not math.isclose(float(actual_near), near_clip_m, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            f"camera near clipping update did not apply: {actual_near!r}"
        )

    if not attrs.get("prim_path") or attrs.get("fov_deg") is not None:
        fov = float(attrs.get("fov_deg", 70.0))
        if not 0.0 < fov < 180.0:
            raise ValueError("camera fov_deg must be between 0 and 180")
        aperture = float(camera.get_horizontal_aperture())
        focal_length = aperture / (2.0 * math.tan(math.radians(fov) / 2.0))
        camera.set_focal_length(focal_length)
        if not math.isclose(
            float(camera.get_focal_length()),
            focal_length,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("camera field-of-view update did not apply")


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
        self._tasks: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._boot_requested = threading.Event()
        self._booted = threading.Event()
        self._boot_error: BaseException | None = None
        self._stop = threading.Event()
        self._sim_thread_id: int | None = None

        self.cfg: SimConfig | None = None
        self.mock = False
        self._sim_app: Any = None
        self.world: Any = None
        self._isaac: Any = None  # lazily-populated namespace of isaac imports
        self._step_callbacks: dict[str, Callable[[float], None]] = {}
        self._mock_sim_time_s = 0.0
        # component name -> (spawn attrs, handle). viam-server rebuilds
        # resources on config change, but prims can't be re-spawned without
        # restarting kit, so handles are cached per component name.
        self._handles: dict[str, tuple[dict[str, Any], Any]] = {}
        self._prop_specs: dict[str, dict[str, Any]] = {}
        self._mock_prop_poses: dict[str, dict[str, Any]] = {}
        self._episode_writer = None
        self._episode_id: str | None = None
        self._cosmos_writer = None
        self._cosmos_output_dir: Path | None = None
        self._cosmos_elapsed_s = 0.0
        self._cosmos_scheduled_count = 0
        self._capture_elapsed_s = 0.0
        self._capture_started_sim_s = 0.0
        self._capture_frame_index = 0
        self._capture_skip_next_step = False
        self._capture_handles: tuple[Any, ...] | None = None
        self._capture_after_render_active = False
        self._capture_last_sim_s = 0.0
        self._cosmos_roles: tuple[str, ...] = ()
        self._nudge_schedule: Any = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def ensure_booted(self, cfg: SimConfig) -> None:
        """Boot the simulator once; later world changes require a restart."""
        if self._booted.is_set():
            if self.cfg != cfg:
                LOGGER.warning(
                    "isaac sim is already running; world config changes require "
                    "restarting the module"
                )
            return
        if self._boot_error is not None:
            raise RuntimeError(
                f"isaac sim failed to boot previously: {self._boot_error}"
            )

        self.cfg = cfg
        self._boot_requested.set()
        if not self._booted.wait(timeout=cfg.boot_timeout):
            raise TimeoutError(f"isaac sim did not boot within {cfg.boot_timeout}s")
        if self._boot_error is not None:
            raise RuntimeError(f"isaac sim failed to boot: {self._boot_error}")

    def _register_props(self, raw_props: Sequence[dict[str, Any]]) -> None:
        specs: dict[str, dict[str, Any]] = {}
        poses: dict[str, dict[str, Any]] = {}
        for raw_prop in raw_props:
            if not raw_prop.get("name"):
                raise ValueError(f"every prop needs a name: {raw_prop}")
            name = _prim_name(str(raw_prop["name"]))
            if name in specs:
                raise ValueError(f"duplicate prop name {name!r}")
            prop = dict(raw_prop)
            prop["name"] = name
            if prop.get("visual_material") is not None:
                _visual_material_properties(prop["visual_material"])
            position = _vector3(prop.get("position", (0.0, 0.0, 0.0)), "position")
            orientation = _quaternion(
                prop.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0)),
                "orientation_wxyz",
            )
            specs[name] = prop
            poses[name] = {
                "position": position,
                "orientation_wxyz": orientation,
            }
        self._prop_specs = specs
        self._mock_prop_poses = poses

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
            for cb in list(self._step_callbacks.values()):
                cb(dt)
            if self.mock:
                # The mock integrates with wall-clock dt, so its sim clock is
                # wall time elapsed since boot.
                self._mock_sim_time_s += dt
                time.sleep(0.01)
            else:
                # One rendered frame plus rendering_dt / physics_dt physics
                # substeps; physics callbacks fire once per substep.
                self.world.step(render=True)
                if self._capture_after_render_active:
                    self._capture_after_render()

        if self._episode_id is not None:
            try:
                self.stop_episode("incomplete")
            except Exception:
                LOGGER.exception("error finalizing active episode during shutdown")

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

    def _start_cosmos_captures(
        self,
        output_dir: Path,
        fixed_camera: "CameraHandle",
        wrist_camera: "CameraHandle",
        fixed_camera_2: Optional["CameraHandle"] = None,
    ) -> Any:
        if self.mock:
            raise RuntimeError("CosmosWriter capture is unavailable in mock mode")
        cameras = {"fixed": fixed_camera, "wrist": wrist_camera}
        if fixed_camera_2 is not None:
            cameras["fixed_2"] = fixed_camera_2
        camera_outputs = {role: output_dir / role for role in cameras}
        for camera_output in camera_outputs.values():
            clip_dir = camera_output / "clip_0000"
            if clip_dir.exists():
                raise FileExistsError(f"CosmosWriter clip already exists: {clip_dir}")

        import carb
        import omni.replicator.core as rep

        carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

        render_products = {}
        writers = {}
        for role, camera in cameras.items():
            camera_prim_path, resolution = camera.cosmos_camera()
            render_products[role] = rep.create.render_product(
                camera_prim_path,
                resolution,
                name=f"ViamCosmos{role.title()}",
            )
            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=str(camera_outputs[role]))
            writers[role] = rep.WriterRegistry.get("CosmosWriter")
            writers[role].initialize(backend=backend, use_instance_id=True)

        render_product_names = {
            role: str(render_product.path).rsplit("/", 1)[-1]
            for role, render_product in render_products.items()
        }

        class DualCosmosWriter(rep.Writer):
            data_structure = "renderProduct"

            def __init__(self) -> None:
                self.annotators = writers["fixed"].annotators

            def write(self, data: dict[str, Any]) -> None:
                shared = {
                    key: value for key, value in data.items() if key != "renderProducts"
                }
                for role, writer in writers.items():
                    payload = dict(shared)
                    payload.update(data["renderProducts"][render_product_names[role]])
                    direct_image_keys = (
                        "rgb",
                        "edges",
                        "depth",
                        writer._shaded_segmentation_annotator,
                    )
                    for key in direct_image_keys:
                        value = payload[key]
                        if isinstance(value, dict) and "data" in value:
                            payload[key] = value["data"]
                    writer.write(payload)

            def on_final_frame(self) -> None:
                for writer in writers.values():
                    writer.on_final_frame()

        writer = DualCosmosWriter()
        attached = False
        self._cosmos_scheduled_count = 0
        try:
            writer.attach(list(render_products.values()), trigger=None)
            attached = True
            writer.schedule_write()
            self._cosmos_scheduled_count = 1
        except BaseException:
            self._cosmos_scheduled_count = 0
            if attached:
                writer.detach()
            for render_product in render_products.values():
                render_product.destroy()
            raise
        return writer, tuple(render_products.values())

    def _cosmos_capture_step(self, step_size: float) -> None:

        capture = self._cosmos_writer
        if capture is None:
            return
        cfg = self.cfg
        if cfg is None:
            return
        period = 1.0 / cfg.capture_rate_hz
        self._cosmos_elapsed_s += float(step_size)
        if self._cosmos_elapsed_s + 1e-9 < period:
            return
        self._cosmos_elapsed_s -= period
        capture[0].schedule_write()
        self._cosmos_scheduled_count += 1

    def _flush_cosmos_capture(self, capture: Any, scheduled_count: int) -> int:
        writer = capture[0]
        if isinstance(scheduled_count, bool) or not isinstance(scheduled_count, int):
            raise RuntimeError(
                "CosmosWriter synchronization failed: scheduled frame count "
                f"must be an integer, got {scheduled_count!r}"
            )
        if scheduled_count < 0:
            raise RuntimeError(
                "CosmosWriter synchronization failed: scheduled frame count "
                f"must be nonnegative, got {scheduled_count}"
            )

        delivered_count = writer.num_written
        if isinstance(delivered_count, bool) or not isinstance(delivered_count, int):
            raise RuntimeError(
                "CosmosWriter synchronization failed: delivered frame count "
                f"must be an integer, got {delivered_count!r}"
            )
        if delivered_count < 0:
            raise RuntimeError(
                "CosmosWriter synchronization failed: delivered frame count "
                f"must be nonnegative, got {delivered_count}"
            )

        pending_count = scheduled_count - delivered_count
        if pending_count not in (0, 1):
            raise RuntimeError(
                "CosmosWriter synchronization failed: expected zero or one pending "
                f"frame, got {pending_count} (scheduled={scheduled_count}, "
                f"delivered={delivered_count})"
            )
        if pending_count == 1:
            self.world.render()

        delivered_count = writer.num_written
        if isinstance(delivered_count, bool) or not isinstance(delivered_count, int):
            raise RuntimeError(
                "CosmosWriter synchronization failed: delivered frame count "
                f"must be an integer, got {delivered_count!r}"
            )
        if delivered_count < 0:
            raise RuntimeError(
                "CosmosWriter synchronization failed: delivered frame count "
                f"must be nonnegative, got {delivered_count}"
            )
        if delivered_count != scheduled_count:
            raise RuntimeError(
                "CosmosWriter synchronization failed after final render: "
                f"scheduled={scheduled_count}, delivered={delivered_count}"
            )
        return delivered_count

    def _finish_cosmos_captures(self, capture: Any) -> None:
        writer, render_products = capture
        first_error = None
        try:
            # CosmosWriter finalization drains its backend queue and encodes all
            # modality videos. Do not stop the global Replicator orchestrator:
            # with capture-on-play enabled, that stops Isaac's timeline and
            # resets articulation state.
            writer.on_final_frame()
        except BaseException as exc:
            first_error = exc
        try:
            writer.detach()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        for render_product in render_products:
            try:
                render_product.destroy()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def start_episode(
        self,
        output_dir: str | None,
        episode_id: str,
        seed: int,
        task: str,
        arm_name: str,
        gripper_name: str,
        fixed_camera_name: str,
        wrist_camera_name: str,
        target: Any,
        cosmos_output_dir: str | None = None,
        fixed_camera_2_name: str = "",
        nudge: Any = None,
    ) -> dict[str, Any]:
        """Start native and/or Cosmos capture on the serialized sim thread.

        With a NudgeConfig, the arm is nudged off its commanded targets while
        the episode records the clean targets plus the applied nudge.
        """

        if not output_dir and not cosmos_output_dir:
            raise ValueError("episode capture requires an output directory")
        if cosmos_output_dir and self.mock:
            raise RuntimeError("CosmosWriter capture is unavailable in mock mode")
        if nudge is not None and self.mock:
            raise RuntimeError("nudges are unavailable in mock mode")
        if nudge is not None and not output_dir:
            raise ValueError("nudges require native episode capture")

        def _start() -> dict[str, Any]:
            if self._episode_id is not None:
                raise RuntimeError("an episode is already being captured")
            cfg = self.cfg
            if cfg is None:
                raise RuntimeError("simulator is not configured")
            sample_rate_hz = cfg.capture_rate_hz
            period = 1.0 / sample_rate_hz
            if not self.mock:
                steps_per_sample = period / cfg.physics_dt
                if not math.isclose(
                    steps_per_sample, round(steps_per_sample), abs_tol=1e-9
                ):
                    raise ValueError(
                        f"physics_dt {cfg.physics_dt:g} cannot produce exact "
                        f"{sample_rate_hz:g} Hz samples"
                    )

            required = [(fixed_camera_name, CameraHandle)]
            if output_dir:
                required = [
                    (arm_name, ArmHandle),
                    (gripper_name, GripperHandle),
                    (fixed_camera_name, CameraHandle),
                    (wrist_camera_name, CameraHandle),
                ]
            elif cosmos_output_dir:
                required.append((wrist_camera_name, CameraHandle))
            if fixed_camera_2_name:
                required.append((fixed_camera_2_name, CameraHandle))
            handles = []
            for name, expected in required:
                entry = self._handles.get(name)
                if entry is None or not isinstance(entry[1], expected):
                    raise ValueError(
                        f"capture component {name!r} is not attached to the sim"
                    )
                handles.append(entry[1])

            writer = None
            cosmos_capture = None
            callback_attached = False
            nudge_callback_attached = False
            cosmos_callback_attached = False
            try:
                if output_dir:
                    from .episode_capture import EpisodeWriter
                    from .episode_schema import RobotDescriptorV2, TargetPoseV2

                    arm_attrs = self._handles[arm_name][0]
                    gripper_attrs = self._handles[gripper_name][0]
                    arm_handle = handles[0]
                    arm_model = str(
                        arm_attrs.get("arm_model")
                        or arm_attrs.get("asset")
                        or arm_attrs.get("usd_path")
                        or arm_name
                    )
                    gripper_model = str(
                        gripper_attrs.get("asset")
                        or gripper_attrs.get("usd_path")
                        or gripper_name
                    )
                    target_pose = TargetPoseV2(
                        prop_name=str(target["prop_name"]),
                        position_m=_vector3(target["position_m"], "target.position_m"),
                        rotation_wxyz=_quaternion(
                            target["rotation_wxyz"], "target.rotation_wxyz"
                        ),
                    )
                    writer = EpisodeWriter(
                        Path(output_dir),
                        episode_id,
                        seed,
                        task,
                        RobotDescriptorV2(
                            arm_component=arm_name,
                            arm_model=arm_model,
                            joint_names=arm_handle.joint_names(),
                            gripper_component=gripper_name,
                            gripper_model=gripper_model,
                            gripper_mode=str(
                                KNOWN_ASSETS.get(gripper_model, {}).get(
                                    "gripper_mode", "position"
                                )
                            ),
                        ),
                        target_pose,
                        sample_rate_hz,
                        fixed_camera_name,
                        wrist_camera_name,
                        fixed_camera_2_name,
                    )
                    self._episode_writer = writer
                    self._capture_handles = tuple(handles)
                    self._capture_elapsed_s = 0.0
                    self._capture_started_sim_s = (
                        self._mock_sim_time_s
                        if self.mock
                        else float(self.world.current_time)
                    )
                    self._capture_frame_index = 0
                    self._capture_skip_next_step = self.mock
                    self._capture_sample(self._capture_started_sim_s)
                    if self.mock:
                        self._step_callbacks["episode_capture"] = self._capture_step
                    elif cfg.capture_after_render:
                        self._capture_last_sim_s = self._capture_started_sim_s
                        self._capture_after_render_active = True
                    else:
                        self.world.add_physics_callback(
                            "episode_capture", self._capture_step
                        )
                    callback_attached = True
                    if nudge is not None:
                        from .nudge import NudgeSchedule

                        self._nudge_schedule = NudgeSchedule(
                            nudge, seed, len(arm_handle.joint_names())
                        )
                        self.world.add_physics_callback(
                            "episode_nudge", self._nudge_step
                        )
                        nudge_callback_attached = True

                if cosmos_output_dir:
                    cosmos_path = Path(cosmos_output_dir).expanduser().resolve()
                    fixed_camera_2 = (
                        self._handles[fixed_camera_2_name][1]
                        if fixed_camera_2_name
                        else None
                    )
                    cosmos_capture = self._start_cosmos_captures(
                        cosmos_path,
                        self._handles[fixed_camera_name][1],
                        self._handles[wrist_camera_name][1],
                        fixed_camera_2,
                    )
                    self._cosmos_roles = tuple(
                        ("fixed", "wrist", "fixed_2")
                        if fixed_camera_2_name
                        else ("fixed", "wrist")
                    )
                    self._cosmos_writer = cosmos_capture
                    self._cosmos_output_dir = cosmos_path
                    self._cosmos_elapsed_s = 0.0
                    self.world.add_physics_callback(
                        "cosmos_capture", self._cosmos_capture_step
                    )
                    cosmos_callback_attached = True

                self._episode_id = episode_id
            except BaseException as exc:
                if cosmos_callback_attached:
                    self.world.remove_physics_callback("cosmos_capture")
                if cosmos_capture is not None:
                    try:
                        self._finish_cosmos_captures(cosmos_capture)
                    except Exception:
                        LOGGER.exception("error cleaning up failed CosmosWriter start")
                if nudge_callback_attached:
                    self.world.remove_physics_callback("episode_nudge")
                    self._clear_nudge()
                if writer is not None:
                    if callback_attached:
                        if self.mock:
                            self._step_callbacks.pop("episode_capture", None)
                        elif self._capture_after_render_active:
                            self._capture_after_render_active = False
                        else:
                            self.world.remove_physics_callback("episode_capture")
                    self._episode_writer = None
                    self._capture_handles = None
                    writer.fail(exc)
                    try:
                        writer.close("incomplete")
                    except RuntimeError:
                        pass
                self._cosmos_writer = None
                self._cosmos_output_dir = None
                self._cosmos_roles = ()
                self._cosmos_elapsed_s = 0.0
                self._cosmos_scheduled_count = 0
                raise

            result: dict[str, Any] = {
                "episode_id": episode_id,
                "sample_rate_hz": sample_rate_hz,
            }
            if writer is not None:
                result["partial_path"] = str(writer.partial_dir)
            if cosmos_output_dir:
                cosmos_path = Path(cosmos_output_dir).expanduser().resolve()
                result["cosmos_partial_paths"] = {
                    role: str(cosmos_path / role / "clip_0000")
                    for role in self._cosmos_roles
                }
            return result

        return self.run(_start, timeout=120.0)

    def active_episode_stage_frame_count(self) -> int | None:
        """Return active native stage length without blocking the sim thread."""

        def _read() -> int | None:
            if self._episode_id is None:
                raise RuntimeError("no episode is being captured")
            writer = self._episode_writer
            if writer is None:
                return None
            return writer.active_stage_frame_count(self._capture_frame_index)

        return self.run(_read, timeout=120.0)

    def mark_episode_stage(self, task: str) -> dict[str, Any]:
        """Start an annotated stage at the next native capture frame.

        Only call this while the simulation is playing.
        """

        def _mark() -> dict[str, Any]:
            if not self.mock and not self.world.is_playing():
                raise RuntimeError(
                    "mark_episode_stage requires the simulation to be playing"
                )
            if self._episode_id is None:
                raise RuntimeError("no episode is being captured")
            writer = self._episode_writer
            if writer is None:
                raise RuntimeError("stage marking requires native episode capture")
            result = writer.mark_stage(task, self._capture_frame_index)
            return {"episode_id": self._episode_id, **result}

        return self.run(_mark, timeout=120.0)

    def _capture_after_render(self) -> None:
        """Advance native capture by the sim time of the update just rendered."""
        now = float(self.world.current_time)
        elapsed = now - self._capture_last_sim_s
        self._capture_last_sim_s = now
        if elapsed > 0.0:
            self._capture_step(elapsed)

    def _capture_step(self, step_size: float) -> None:

        writer = self._episode_writer
        if writer is None or writer.failed:
            return
        if self._capture_skip_next_step:
            self._capture_skip_next_step = False
            return
        cfg = self.cfg
        if cfg is None:
            writer.fail(RuntimeError("simulator is not configured"))
            return
        period = 1.0 / cfg.capture_rate_hz
        self._capture_elapsed_s += float(step_size)
        if self._capture_elapsed_s + 1e-9 < period:
            return
        if self._capture_elapsed_s >= 2.0 * period:
            writer.fail(
                RuntimeError(
                    f"simulator skipped a {cfg.capture_rate_hz:g} Hz capture boundary"
                )
            )
            return
        self._capture_elapsed_s -= period
        sim_time_s = self._capture_started_sim_s + self._capture_frame_index * period
        try:
            self._capture_sample(sim_time_s)
        except BaseException as exc:
            writer.fail(exc)

    def _nudge_step(self, step_size: float) -> None:
        schedule = self._nudge_schedule
        handles = self._capture_handles
        if schedule is None or handles is None:
            return
        try:
            handles[0].set_nudge(schedule.step(float(step_size)))
        except BaseException as exc:
            if self._episode_writer is not None:
                self._episode_writer.fail(exc)

    def _clear_nudge(self) -> None:
        handles = self._capture_handles
        if self._nudge_schedule is not None and handles is not None:
            handles[0].set_nudge([0.0] * len(handles[0].joint_names()))
        self._nudge_schedule = None

    def _capture_sample(self, sim_time_s: float) -> None:
        from .episode_schema import EpisodeSampleV2

        writer = self._episode_writer
        handles = self._capture_handles
        if writer is None or handles is None:
            return
        arm, gripper, fixed_camera, wrist_camera, *optional_cameras = handles
        positions, velocities, targets, ee_position, ee_rotation = arm.capture_state()
        nudge = arm.nudge_offsets() or [0.0] * len(targets)
        gripper_command, gripper_state = gripper.capture_state()
        fixed_rgb = fixed_camera.capture_rgb()
        wrist_rgb = wrist_camera.capture_rgb()
        fixed_2_rgb = optional_cameras[0].capture_rgb() if optional_cameras else None
        frame_index = self._capture_frame_index
        writer.submit(
            EpisodeSampleV2(
                frame_index=frame_index,
                sim_time_s=sim_time_s,
                joint_position_rad=tuple(positions),
                joint_velocity_rad_s=tuple(velocities),
                joint_target_rad=tuple(targets),
                ee_position_m=tuple(ee_position),
                ee_rotation_wxyz=tuple(ee_rotation),
                gripper_command=gripper_command,
                gripper_state=gripper_state,
                joint_target_nudge_rad=tuple(nudge),
            ),
            fixed_rgb,
            wrist_rgb,
            fixed_2_rgb,
        )
        self._capture_frame_index += 1

    def stop_episode(self, outcome: str) -> dict[str, Any]:
        def _detach():
            if self._episode_id is None:
                raise RuntimeError("no episode is being captured")
            episode_id = self._episode_id
            writer = self._episode_writer
            cosmos_capture = self._cosmos_writer
            cosmos_output_dir = self._cosmos_output_dir
            cosmos_scheduled_count = self._cosmos_scheduled_count
            cosmos_roles = self._cosmos_roles

            # Stop both physics-driven producers before the render-only update
            # that may be needed to deliver the final scheduled Cosmos frame.
            if writer is not None:
                if self.mock:
                    self._step_callbacks.pop("episode_capture", None)
                elif self._capture_after_render_active:
                    self._capture_after_render_active = False
                else:
                    self.world.remove_physics_callback("episode_capture")
            if self._nudge_schedule is not None:
                self.world.remove_physics_callback("episode_nudge")
                self._clear_nudge()
            if cosmos_capture is not None:
                self.world.remove_physics_callback("cosmos_capture")

            cosmos_delivered_count = None
            cosmos_error = None
            cosmos_synchronization_failed = False
            if cosmos_capture is not None:
                try:
                    cosmos_delivered_count = self._flush_cosmos_capture(
                        cosmos_capture, cosmos_scheduled_count
                    )
                except BaseException as exc:
                    cosmos_error = exc
                    cosmos_synchronization_failed = True
                try:
                    self._finish_cosmos_captures(cosmos_capture)
                except BaseException as exc:
                    if cosmos_error is None:
                        cosmos_error = exc

            self._episode_id = None
            self._episode_writer = None
            self._capture_handles = None
            self._capture_skip_next_step = False
            self._cosmos_writer = None
            self._cosmos_output_dir = None
            self._cosmos_roles = ()
            self._cosmos_elapsed_s = 0.0
            self._cosmos_scheduled_count = 0
            return (
                episode_id,
                writer,
                cosmos_output_dir,
                cosmos_scheduled_count,
                cosmos_roles,
                cosmos_delivered_count,
                cosmos_error,
                cosmos_synchronization_failed,
            )

        (
            episode_id,
            writer,
            cosmos_output_dir,
            cosmos_scheduled_count,
            cosmos_roles,
            cosmos_delivered_count,
            cosmos_error,
            cosmos_synchronization_failed,
        ) = self.run(_detach, timeout=120.0)
        result: dict[str, Any] = {"episode_id": episode_id, "outcome": outcome}
        native_error = None
        native_sample_count = None
        native_final_path = None
        native_stage_start_frames = (0,)
        if writer is not None:
            try:
                native_final_path = writer.close(outcome)
                native_sample_count = writer.sample_count
                native_stage_start_frames = writer.stage_start_frames() or (0,)
                if isinstance(native_sample_count, bool) or not isinstance(
                    native_sample_count, int
                ):
                    raise RuntimeError(
                        "native sample_count must be an integer, got "
                        f"{native_sample_count!r}"
                    )
                if native_sample_count < 0:
                    raise RuntimeError(
                        "native sample_count must be nonnegative, got "
                        f"{native_sample_count}"
                    )
                result.update(
                    path=str(native_final_path), sample_count=native_sample_count
                )
            except BaseException as exc:
                native_error = exc
        if cosmos_error is not None:
            if cosmos_synchronization_failed:
                detail = str(cosmos_error)
                if detail.startswith("CosmosWriter synchronization failed"):
                    raise RuntimeError(detail) from cosmos_error
                raise RuntimeError(
                    f"CosmosWriter synchronization failed: {detail}"
                ) from cosmos_error
            raise RuntimeError("CosmosWriter finalization failed") from cosmos_error
        if native_error is not None:
            raise RuntimeError("native episode finalization failed") from native_error
        if cosmos_output_dir is not None:
            if cosmos_scheduled_count <= 0:
                raise RuntimeError("CosmosWriter scheduled no frames")
            if cosmos_delivered_count != cosmos_scheduled_count:
                raise RuntimeError(
                    "CosmosWriter synchronization failed: "
                    f"scheduled={cosmos_scheduled_count}, "
                    f"delivered={cosmos_delivered_count}"
                )
            if writer is not None and (
                native_sample_count != cosmos_scheduled_count
                or native_sample_count != cosmos_delivered_count
            ):
                raise RuntimeError(
                    f"CosmosWriter scheduled frame count {cosmos_scheduled_count} "
                    f"does not match native sample_count {native_sample_count} "
                    f"and delivered frame count {cosmos_delivered_count}"
                )

            from .cosmos_capture import finalize_cosmos_videos

            try:
                cfg = self.cfg
                if cfg is None:
                    raise RuntimeError("simulator is not configured")
                finalize_cosmos_videos(
                    cosmos_output_dir,
                    cosmos_scheduled_count,
                    sample_rate_hz=cfg.capture_rate_hz,
                    roles=cosmos_roles,
                    stage_start_frames=native_stage_start_frames,
                )
            except Exception as exc:
                raise RuntimeError(
                    "CosmosWriter video finalization failed for "
                    f"{cosmos_output_dir}: {exc}"
                ) from exc
            result["cosmos_paths"] = {
                role: str(cosmos_output_dir / role / "clip_0000")
                for role in cosmos_roles
            }
            result["cosmos_output_dir"] = str(cosmos_output_dir)
        return result

    # ------------------------------------------------------------------
    # boot
    # ------------------------------------------------------------------

    def _boot(self) -> None:
        cfg = self.cfg
        assert cfg is not None
        self._register_props(cfg.props)
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

        if cfg.capture_after_render:
            # Isaac Lab's settings for synchronous camera data ("Avoids frame
            # offset issue"): each update waits for its render to finish, so
            # annotators return the frame of the state just simulated.
            sys.argv.extend(
                [
                    "--/app/asyncRendering=false",
                    "--/omni/replicator/asyncRendering=false",
                    "--/app/hydraEngine/waitIdle=true",
                    "--/app/renderer/waitIdle=true",
                    "--/app/updateOrder/checkForHydraRenderComplete=1000",
                ]
            )

        streaming = cfg.livestream and cfg.headless
        launch: dict[str, Any] = {"headless": cfg.headless}
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
                    self._sim_app.set_setting(
                        "/app/livestream/publicEndpointAddress", ip
                    )
                # 5.0 ships the streaming service under omni.services; 4.5 and
                # older only have the kit extension
                for ext in (
                    "omni.services.livestream.nvcf",
                    "omni.kit.livestream.webrtc",
                ):
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

        rendering_dt = cfg.resolved_rendering_dt()
        LOGGER.info(
            "simulation dt: physics %.5fs, rendering %.5fs (%d physics substeps "
            "per rendered frame), capture %.5fs",
            cfg.physics_dt,
            rendering_dt,
            round(rendering_dt / cfg.physics_dt),
            1.0 / cfg.capture_rate_hz,
        )
        self.world = self._isaac.World(
            physics_dt=cfg.physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=1.0,
        )
        if cfg.dome_light:
            import omni.usd
            from pxr import Sdf, UsdLux

            stage = omni.usd.get_context().get_stage()
            dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
            dome_light.CreateIntensityAttr(1000.0)
        if not cfg.usd_stage:
            if cfg.isaac_default_environment:
                self.world.scene.add_default_ground_plane()
            else:
                self._add_fallback_lighting()
        for prop in self._prop_specs.values():
            self._spawn_prop(prop)
        self.world.reset()
        LOGGER.info("isaac sim world ready")

    def _add_fallback_lighting(self) -> None:
        from pxr import Sdf, UsdLux

        stage = self.world.stage
        distant = UsdLux.DistantLight.Define(
            stage, Sdf.Path("/World/DefaultDistantLight")
        )
        distant.CreateIntensityAttr(3000.0)
        distant.CreateAngleAttr(0.53)
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DefaultDomeLight"))
        dome.CreateIntensityAttr(500.0)
        LOGGER.info("added fallback DistantLight + DomeLight")

    def _spawn_prop(self, prop: dict[str, Any]) -> None:
        """Add a configured prop to the scene (runs on the sim thread,
        before the initial world.reset)."""
        import numpy as np

        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = _prim_name(str(prop["name"]))
        prim_path = f"/World/{name}"
        position = list(_vector3(prop.get("position", (0.0, 0.0, 0.0)), "position"))
        kind = str(prop.get("type", "cube"))

        if kind == "usd":
            usd_path = prop.get("usd_path")
            if not usd_path:
                raise ValueError(f"prop {name}: type 'usd' needs usd_path")
            if self._usd_exists(usd_path) is False:
                raise ValueError(f"prop {name}: usd not found: {usd_path}")
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            pose = {"position": position}
            if prop.get("orientation_wxyz") is not None:
                pose["orientation"] = list(
                    _quaternion(prop["orientation_wxyz"], "orientation_wxyz")
                )
            self._isaac.SingleXFormPrim(prim_path).set_world_pose(**pose)
            return

        if kind != "cube":
            raise ValueError(f"prop {name}: unknown type {kind!r} (cube or usd)")

        kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            position=np.array(position),
            size=float(prop.get("size", 0.05)),
        )
        if prop.get("orientation_wxyz") is not None:
            kwargs["orientation"] = np.array(
                _quaternion(prop["orientation_wxyz"], "orientation_wxyz")
            )
        if prop.get("scale") is not None:
            kwargs["scale"] = np.array([float(v) for v in prop["scale"]])
        if prop.get("color") is not None:
            kwargs["color"] = np.array([float(v) for v in prop["color"]])
        static_friction = float(prop.get("static_friction", 1.0))
        dynamic_friction = float(prop.get("dynamic_friction", 1.0))
        if static_friction < 0.0 or dynamic_friction < 0.0:
            raise ValueError(f"prop {name}: friction must be non-negative")
        kwargs["physics_material"] = self._isaac.PhysicsMaterial(
            prim_path=f"/World/Physics_Materials/{name}",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=0.0,
        )
        cls = (
            self._isaac.FixedCuboid if prop.get("fixed") else self._isaac.DynamicCuboid
        )
        spawned = cls(**kwargs)
        spawned.set_rest_offset(GRASP_REST_OFFSET_M)
        spawned.set_contact_offset(GRASP_CONTACT_OFFSET_M)
        self.world.scene.add(spawned)
        if prop.get("visual_material") is not None:
            self._apply_visual_material(prim_path, str(prop["visual_material"]))

    def _apply_visual_material(self, prim_path: str, preset: str) -> None:
        from pxr import Gf, Sdf, UsdShade

        properties = _visual_material_properties(preset)
        prim = self._isaac.get_prim_at_path(prim_path)
        if prim is None or not prim.IsValid():
            raise ValueError(f"cannot bind visual material to missing prim {prim_path}")
        material_path = (
            f"/World/Looks/{_prim_name(prim_path.rsplit('/', 1)[-1])}_{preset}"
        )
        material = UsdShade.Material.Define(prim.GetStage(), material_path)
        shader = UsdShade.Shader.Define(prim.GetStage(), f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*properties["diffuse_color"])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            properties["roughness"]
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            properties["metallic"]
        )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

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

    def is_playing(self) -> bool:
        """Return whether the simulation timeline is playing."""
        if self.mock:
            return True
        self._require_booted()
        return self.run(lambda: bool(self.world.is_playing()))

    def reset(self) -> None:
        self._require_booted()
        if not self.mock:
            self.run(lambda: self.world.reset())

    def sim_time_s(self) -> float:
        """Simulator seconds since boot; safe to call from any thread."""
        if self.mock:
            return self._mock_sim_time_s
        self._require_booted()
        return self.run(lambda: float(self.world.current_time))

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "booted": self._booted.is_set(),
            "mock": self.mock,
            "error": str(self._boot_error) if self._boot_error else "",
        }
        if self._booted.is_set() and not self.mock:
            out["playing"] = self.is_playing()
            out["sim_time"] = self.sim_time_s()
        return out

    def _prop_spec(self, name: str) -> tuple[str, dict[str, Any]]:
        prim_name = _prim_name(name)
        spec = self._prop_specs.get(prim_name)
        if spec is None:
            raise ValueError(f"unknown prop {name!r}; have {sorted(self._prop_specs)}")
        return prim_name, spec

    def _set_prop_pose(
        self,
        name: str,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        if self.mock:
            pose = {"position": position, "orientation_wxyz": orientation}
            self._mock_prop_poses[name] = pose
            return pose

        import numpy as np

        scene_object = self.world.scene.get_object(name)
        target = (
            scene_object
            if scene_object is not None
            else self._isaac.SingleXFormPrim(f"/World/{name}")
        )
        target.set_world_pose(position=list(position), orientation=list(orientation))
        for setter_name in ("set_linear_velocity", "set_angular_velocity"):
            setter = getattr(scene_object, setter_name, None)
            if setter is not None:
                setter(np.zeros(3))
        return {"position": position, "orientation_wxyz": orientation}

    def get_prop_pose(self, name: str) -> dict[str, Any]:
        self._require_booted()
        prim_name, _spec = self._prop_spec(name)

        def read() -> dict[str, Any]:
            if self.mock:
                return dict(self._mock_prop_poses[prim_name])
            scene_object = self.world.scene.get_object(prim_name)
            target = (
                scene_object
                if scene_object is not None
                else self._isaac.SingleXFormPrim(f"/World/{prim_name}")
            )
            position, orientation = target.get_world_pose()
            return {
                "position": tuple(float(v) for v in position),
                "orientation_wxyz": tuple(float(v) for v in orientation),
            }

        pose = self.run(read)
        return {
            "position": list(pose["position"]),
            "orientation_wxyz": list(pose["orientation_wxyz"]),
        }

    def set_prop_pose(
        self,
        name: str,
        position: Sequence[float],
        orientation: Sequence[float],
    ) -> dict[str, Any]:
        self._require_booted()
        prim_name, _spec = self._prop_spec(name)
        pos = _vector3(position, "position")
        ori = _quaternion(orientation, "orientation_wxyz")

        def apply() -> dict[str, Any]:
            result = self._set_prop_pose(prim_name, pos, ori)
            return {
                "position": list(result["position"]),
                "orientation_wxyz": list(result["orientation_wxyz"]),
            }

        return self.run(apply)

    def park_props(self, names: Sequence[str]) -> dict[str, Any]:
        self._require_booted()
        registered = [self._prop_spec(str(name)) for name in names]

        def park() -> dict[str, Any]:
            poses: dict[str, Any] = {}
            for name, spec in registered:
                position = _vector3(spec.get("position", (0.0, 0.0, 0.0)), "position")
                orientation = _quaternion(
                    spec.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0)),
                    "orientation_wxyz",
                )
                pose = self._set_prop_pose(name, position, orientation)
                poses[name] = {
                    "position": list(pose["position"]),
                    "orientation_wxyz": list(pose["orientation_wxyz"]),
                }
            return {"poses": poses}

        return self.run(park)

    def add_usd_reference(
        self,
        usd_path: str,
        prim_path: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
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

    _CAMERA_MUTABLE_KEYS = frozenset(
        {
            "position",
            "target",
            "orientation_wxyz",
            "orientation_rpy_deg",
            "local_position",
            "local_orientation_wxyz",
            "local_orientation_rpy_deg",
            "local_camera_axes",
            "fov_deg",
            "near_clip_m",
        }
    )

    def _cached_handle(
        self, kind: str, name: str, attrs: dict[str, Any], factory: Callable[[], Any]
    ) -> Any:
        if name in self._handles:
            old_attrs, handle = self._handles[name]
            strip = lambda a: {
                k: v for k, v in a.items() if k not in self._RUNTIME_KEYS
            }
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

    def _usd_exists(self, path: str) -> bool | None:
        """True/False if we can check, None if omni.client is unavailable."""
        client = getattr(self._isaac, "client", None)
        if client is None:
            return None
        try:
            result, _ = client.stat(path)
            return result == client.Result.OK
        except Exception:
            return None

    def _resolve_usd(self, attrs: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Return (absolute usd path or None, known-asset metadata)."""
        meta: dict[str, Any] = {}
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

    def create_arm(self, name: str, attrs: dict[str, Any]) -> "ArmHandle":
        self._require_booted()
        if self.mock:
            factory = lambda: MockArmHandle(self, name, attrs)
        else:
            factory = lambda: self.run(
                lambda: self._create_arm_isaac(name, attrs), timeout=120.0
            )
        return self._cached_handle("arm", name, attrs, factory)

    def _create_arm_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacArmHandle":
        from .spatial import to_vec3

        usd, meta = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        if usd:
            self._isaac.add_reference_to_stage(usd_path=usd, prim_path=prim_path)
            self._require_prim(prim_path)
            asset_prim = self._isaac.get_prim_at_path(prim_path)
            for variant_set, selection in meta.get("variants", {}).items():
                variants = asset_prim.GetVariantSets().GetVariantSet(variant_set)
                if not variants.SetVariantSelection(selection):
                    raise ValueError(
                        f"asset {attrs.get('asset')!r}: could not select "
                        f"{variant_set}={selection!r}"
                    )

        # Set the base pose directly on the USD prim. The `position=` /
        # `orientation=` kwargs on SingleArticulation don't persist across the
        # world.reset() below when another articulation is already in the
        # scene, so every arm silently ends up at the world origin and their
        # bodies penetrate. Writing to USD persists across resets.
        position = to_vec3(attrs.get("position"))
        orientation = (
            [float(v) for v in attrs["orientation_wxyz"]]
            if attrs.get("orientation_wxyz") is not None
            else None
        )
        pose_kwargs: dict[str, Any] = {"position": list(position)}
        if orientation is not None:
            pose_kwargs["orientation"] = orientation
        try:
            self._isaac.SingleXFormPrim(prim_path).set_world_pose(**pose_kwargs)
        except Exception:
            LOGGER.exception("failed to set base pose for %s", name)

        art = self._isaac.SingleArticulation(prim_path=prim_path, name=name)
        self.world.scene.add(art)
        _reset_articulation_at_pose(self.world, art, position, orientation)

        ee = None
        ee_path = attrs.get("end_effector_prim")
        if not ee_path and meta.get("end_effector_link"):
            ee_path = f"{prim_path}/{meta['end_effector_link']}"
        if ee_path:
            ee = self._isaac.SingleXFormPrim(ee_path)
        return IsaacArmHandle(
            self,
            art,
            ee,
            prim_path,
            meta.get("joint_names"),
            is_moving_velocity_threshold_rad_s=float(
                attrs.get(
                    "is_moving_velocity_threshold_rad_s",
                    DEFAULT_ARM_IS_MOVING_VELOCITY_THRESHOLD_RAD_S,
                )
            ),
            max_joint_deg_s=(
                float(attrs["max_joint_deg_s"]) if "max_joint_deg_s" in attrs else None
            ),
            max_joint_deg_s2=(
                float(attrs["max_joint_deg_s2"])
                if "max_joint_deg_s2" in attrs
                else None
            ),
        )

    def create_gripper(self, name: str, attrs: dict[str, Any]) -> "GripperHandle":
        """Attach a supported gripper to an already-created arm articulation."""
        self._require_booted()
        gripper_attrs = dict(attrs)
        gripper_attrs.setdefault("asset", "robotiq_2f_85")
        asset = str(gripper_attrs["asset"])
        meta = KNOWN_ASSETS.get(asset)
        if meta is None or meta.get("kind") != "gripper":
            known = sorted(
                name
                for name, value in KNOWN_ASSETS.items()
                if value.get("kind") == "gripper"
            )
            raise ValueError(f"unknown gripper asset {asset!r}; known: {known}")

        arm_name = str(gripper_attrs.get("arm", ""))
        arm_entry = self._handles.get(arm_name)
        if arm_entry is None or not isinstance(arm_entry[1], ArmHandle):
            raise ValueError(
                f"gripper {name!r}: arm {arm_name!r} is not attached to the sim"
            )
        arm_attrs, arm_handle = arm_entry
        integrated_arm_asset = meta.get("integrated_arm_asset")
        if integrated_arm_asset and arm_attrs.get("asset") != integrated_arm_asset:
            raise ValueError(
                f"gripper {asset!r} requires arm asset {integrated_arm_asset!r}, "
                f"got {arm_attrs.get('asset')!r}"
            )
        if self.mock:
            factory = lambda: MockGripperHandle(name, gripper_attrs, arm_handle)
        elif integrated_arm_asset:
            factory = lambda: self.run(
                lambda: self._create_integrated_gripper_isaac(
                    name, gripper_attrs, arm_handle, meta
                )
            )
        else:
            factory = lambda: self.run(
                lambda: self._create_gripper_isaac(
                    name, gripper_attrs, arm_attrs, arm_handle
                ),
                timeout=120.0,
            )
        return self._cached_handle("gripper", name, gripper_attrs, factory)

    def _create_integrated_gripper_isaac(
        self,
        name: str,
        attrs: dict[str, Any],
        arm: "ArmHandle",
        meta: dict[str, Any],
    ) -> "IsaacGripperHandle":
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        if not isinstance(arm, IsaacArmHandle):
            raise ValueError(f"gripper {name!r}: configured arm is not an Isaac arm")
        layout_name = str(attrs.get("jaw_layout", "wide"))
        try:
            layout_signs = LITE6_JAW_LAYOUTS[layout_name]
        except KeyError:
            raise ValueError(
                f"gripper {name!r}: unknown Lite6 jaw_layout {layout_name!r}; "
                f"known: {sorted(LITE6_JAW_LAYOUTS)}"
            ) from None
        self._require_prim(f"{arm._prim_path}/gripper/uflite_gripper_link")
        gripper_root = self._isaac.get_prim_at_path(f"{arm._prim_path}/gripper")
        finger_material = self._isaac.PhysicsMaterial(
            prim_path=f"/World/Physics_Materials/{_prim_name(name)}_fingers",
            static_friction=float(attrs.get("static_friction", 2.0)),
            dynamic_friction=float(attrs.get("dynamic_friction", 2.0)),
            restitution=0.0,
        )
        stage = gripper_root.GetStage()
        collider_specs = (
            (
                "base",
                (0.0, 0.0078000005, 0.0034),
                (0.010000002, 0.0078000005, 0.0034),
            ),
            (
                "outer_tip",
                (0.0, 0.0134000005, 0.012050001),
                (0.010000002, 0.0022000005, 0.014750002),
            ),
        )
        mimic_joint = stage.GetPrimAtPath(
            f"{arm._prim_path}/gripper/joints/finger_joint2"
        )
        for attribute_name, value in (
            (
                "physxMimicJoint:rotY:naturalFrequency",
                LITE6_MIMIC_NATURAL_FREQUENCY,
            ),
            ("physxMimicJoint:rotY:dampingRatio", LITE6_MIMIC_DAMPING_RATIO),
        ):
            attribute = mimic_joint.GetAttribute(attribute_name)
            if not attribute:
                raise ValueError(
                    f"gripper {name!r}: mimic joint is missing {attribute_name!r}"
                )
            attribute.Set(value)

        for finger_index, natural_y_sign in ((1, 1.0), (2, -1.0)):
            y_sign = layout_signs[finger_index - 1]
            reversed_bracket = y_sign != natural_y_sign
            mount_offset_y = (
                -y_sign * LITE6_JAW_BASE_WIDTH_M if reversed_bracket else 0.0
            )
            finger_path = f"{arm._prim_path}/gripper/uflite_finger{finger_index}"
            visuals_path = f"{finger_path}/visuals"
            self._require_prim(visuals_path)
            visuals = UsdGeom.Xformable(self._isaac.get_prim_at_path(visuals_path))
            translate_op = _get_or_add_xform_op(
                visuals, UsdGeom.XformOp.TypeTranslate, visuals.AddTranslateOp
            )
            rotate_op = _get_or_add_xform_op(
                visuals, UsdGeom.XformOp.TypeRotateZ, visuals.AddRotateZOp
            )
            translate_op.Set(Gf.Vec3d(0.0, mount_offset_y, 0.0))
            rotate_op.Set(180.0 if reversed_bracket else 0.0)
            visuals.SetXformOpOrder([translate_op, rotate_op])
            authored_collision_path = f"{finger_path}/collisions"
            self._require_prim(authored_collision_path)
            authored_collision = self._isaac.get_prim_at_path(authored_collision_path)
            authored_collision_api = UsdPhysics.CollisionAPI(authored_collision)
            authored_collision_api.CreateCollisionEnabledAttr(False)

            for collider_name, center, half_extents in collider_specs:
                proxy = UsdGeom.Cube.Define(
                    stage, f"{finger_path}/viam_collision/{collider_name}"
                )
                proxy.CreateSizeAttr(1.0)
                _get_or_add_xform_op(
                    proxy, UsdGeom.XformOp.TypeTranslate, proxy.AddTranslateOp
                ).Set(
                    Gf.Vec3d(center[0], mount_offset_y + y_sign * center[1], center[2])
                )
                _get_or_add_xform_op(
                    proxy, UsdGeom.XformOp.TypeScale, proxy.AddScaleOp
                ).Set(Gf.Vec3d(*half_extents))
                proxy.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                collision_prim = proxy.GetPrim()
                UsdPhysics.CollisionAPI.Apply(collision_prim)
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
                physx_collision.CreateRestOffsetAttr(GRASP_REST_OFFSET_M)
                physx_collision.CreateContactOffsetAttr(GRASP_CONTACT_OFFSET_M)
                UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(
                    finger_material.material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
        dynamic_prop_paths = [
            f"/World/{prop_name}"
            for prop_name, spec in self._prop_specs.items()
            if not spec.get("fixed")
        ]
        contact_view = None
        if dynamic_prop_paths:
            contact_view = self._isaac.RigidContactView(
                prim_paths_expr=[
                    f"{arm._prim_path}/gripper/uflite_finger1/viam_collision/outer_tip",
                    f"{arm._prim_path}/gripper/uflite_finger2/viam_collision/outer_tip",
                ],
                filter_paths_expr=[dynamic_prop_paths, dynamic_prop_paths],
                name=f"{_prim_name(name)}_contacts",
            )
        # Editing collision prims invalidates Isaac's live tensor view. Rebuild
        # it before reading DOF names or limits from the articulation wrapper.
        self.world.reset()
        if contact_view is not None:
            contact_view.initialize()
        arm.refresh_dofs()
        articulation = arm._art
        drive_joint = str(meta["drive_joint"])
        names = list(articulation.dof_names)
        if drive_joint not in names:
            raise ValueError(
                f"gripper drive joint {drive_joint!r} not found in articulation DOFs: {names}"
            )
        drive_index = names.index(drive_joint)
        lower, upper = _dof_limit_pair(articulation, drive_index)
        bounds = {"lower": lower, "upper": upper}
        open_position = bounds[str(meta["open_limit"])]
        closed_position = bounds[str(meta["closed_limit"])]
        if not all(math.isfinite(value) for value in (open_position, closed_position)):
            raise ValueError(
                f"gripper {name!r}: jaw limits are not finite: {lower}..{upper}"
            )

        return IsaacGripperHandle(
            self,
            articulation,
            drive_joint,
            open_position,
            closed_position,
            float(attrs.get("holding_tolerance_m", meta["holding_tolerance"])),
            position_unit=str(meta["position_unit"]),
            minimum_grasp_closure=float(meta.get("minimum_grasp_closure", 0.0)),
            binary=True,
            coupled_drive_joints=meta.get("coupled_drive_joints", ()),
            drive_stiffness=float(meta.get("drive_stiffness", GRIPPER_DRIVE_STIFFNESS)),
            drive_damping=float(meta.get("drive_damping", GRIPPER_DRIVE_DAMPING)),
            contact_view=contact_view,
            contact_dt=float(
                self.cfg.physics_dt if self.cfg is not None else 1.0 / 60.0
            ),
        )

    def _create_gripper_isaac(
        self,
        name: str,
        attrs: dict[str, Any],
        arm_attrs: dict[str, Any],
        arm: "ArmHandle",
    ) -> "IsaacGripperHandle":
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics

        from .spatial import quat_from_euler_deg, to_vec3

        if not isinstance(arm, IsaacArmHandle):
            raise ValueError(f"gripper {name!r}: configured arm is not an Isaac arm")

        usd, meta = self._resolve_usd(attrs)
        if usd is None:
            raise ValueError(f"gripper {name!r}: no USD asset resolved")
        try:
            from pxr import UsdUtils
        except ImportError:
            UsdUtils = None
        usd, preparation = self._prepared_asset_layer(Sdf, UsdUtils, usd)
        LOGGER.info("gripper %r asset preparation: %s", name, preparation)

        gripper_prim_path = f"{arm._prim_path}/Gripper"
        self._isaac.add_reference_to_stage(usd_path=usd, prim_path=gripper_prim_path)
        gripper_root = self._isaac.get_prim_at_path(gripper_prim_path)
        if gripper_root is None or not gripper_root.IsValid():
            raise ValueError(
                f"gripper {name!r}: {usd} did not compose at {gripper_prim_path}"
            )

        composition = _describe_composition(Usd, Sdf, gripper_root, usd)
        if not composition["children"] and composition["layer_root_prims"]:
            if not composition["layer_default_prim"]:
                root_path = composition["layer_root_prims"][0]
                refs = gripper_root.GetReferences()
                refs.ClearReferences()
                refs.AddReference(Sdf.Reference(usd, Sdf.Path(root_path)))
                composition = _describe_composition(Usd, Sdf, gripper_root, usd)
        if not composition["children"]:
            raise ValueError(
                f"gripper {name!r}: asset composed no children at {gripper_prim_path}; "
                f"details: {composition}"
            )

        removed = _remove_articulation_roots(Usd, UsdPhysics, PhysxSchema, gripper_root)
        LOGGER.info("gripper %r removed nested articulation roots: %s", name, removed)

        base_link = next(
            (
                prim
                for prim in _prim_range(Usd, gripper_root)
                if prim.GetName() == "base_link"
            ),
            None,
        )
        if base_link is None:
            raise ValueError(f"gripper {name!r}: base_link not found in {usd}")

        parent_prim = attrs.get("parent_prim") or f"{arm._prim_path}/wrist_3_link"
        self._require_prim(str(parent_prim))
        local_position = to_vec3(attrs.get("local_position"))
        r, p, y = to_vec3(attrs.get("local_orientation_rpy_deg"))
        local_quat = quat_from_euler_deg(r, p, y)
        fixed_joint = UsdPhysics.FixedJoint.Define(
            gripper_root.GetStage(), f"{gripper_prim_path}/WristFixedJoint"
        )
        fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(str(parent_prim))])
        fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(str(base_link.GetPath()))])
        fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(*local_position))
        fixed_joint.CreateLocalRot0Attr(
            Gf.Quatf(local_quat[0], Gf.Vec3f(*local_quat[1:]))
        )
        fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        pads = _pad_collision_status(Usd, UsdPhysics, gripper_root)
        if not any(in_subtree for _path, _self, in_subtree in pads):
            raise ValueError(
                f"gripper {name!r}: Robotiq fingertip collision geometry did not "
                f"compose; pads seen: {pads}; preparation: {preparation}"
            )

        # The wrapper was initialized before the gripper DOFs existed. Isaac's
        # reset reuses its cached six-DOF defaults unless the view is invalidated.
        articulation_view = arm._art._articulation_view
        articulation_view._is_initialized = False
        articulation_view._physics_view = None
        articulation_view._default_joints_state = None
        self.world.reset()
        arm.refresh_dofs()
        articulation = arm._art

        drive_joint = str(meta.get("drive_joint", "finger_joint"))
        names = list(articulation.dof_names)
        if drive_joint not in names:
            raise ValueError(
                f"gripper drive joint {drive_joint!r} not found in articulation DOFs: {names}"
            )
        drive_index = names.index(drive_joint)
        open_deg = float(attrs.get("open_deg", meta.get("open_deg", 0.0)))
        closed_deg = float(attrs.get("closed_deg", meta.get("closed_deg", 45.0)))
        try:
            lower, upper = _dof_limit_pair(articulation, drive_index)
            if "open_deg" not in attrs and math.isfinite(lower):
                open_deg = math.degrees(lower)
            if "closed_deg" not in attrs and math.isfinite(upper):
                closed_deg = math.degrees(upper)
        except Exception:
            LOGGER.warning("gripper %r: could not read authored jaw limits", name)
        if closed_deg <= open_deg:
            raise ValueError(
                f"gripper {name!r}: closed_deg must exceed open_deg, got "
                f"{open_deg}..{closed_deg}"
            )
        return IsaacGripperHandle(
            self,
            articulation,
            drive_joint,
            math.radians(open_deg),
            math.radians(closed_deg),
            math.radians(float(attrs.get("holding_tolerance_deg", 2.0))),
            binary=str(meta.get("gripper_mode", "position")) == "binary",
        )

    def _prepared_asset_layer(
        self, sdf: Any, usd_utils: Any, usd: str
    ) -> tuple[str, dict[str, Any]]:
        report: dict[str, Any] = {
            "source": usd,
            "applied": [],
            "missing": [],
            "anchored": [],
        }
        if usd_utils is None:
            report["reason"] = "pxr.UsdUtils unavailable"
            return usd, report
        prepared = self._prepare_layer(
            sdf, usd_utils, usd, report, {}, ASSET_PREPARE_MAX_DEPTH
        )
        report.setdefault(
            "reason",
            "unchanged" if prepared == usd else "prepared",
        )
        return prepared, report

    def _prepare_layer(
        self,
        sdf: Any,
        usd_utils: Any,
        usd: str,
        report: dict[str, Any],
        prepared_by_source: dict[str, str],
        depth_left: int,
    ) -> str:
        if usd in prepared_by_source:
            return prepared_by_source[usd]
        prepared_by_source[usd] = usd
        stem = usd.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        digest = hashlib.sha1(usd.encode()).hexdigest()[:12]
        local = asset_cache_dir() / f"{stem}-{digest}-v{ASSET_LAYER_CACHE_VERSION}.usd"
        if local.exists():
            report["reason"] = "cached"
            prepared_by_source[usd] = str(local)
            return str(local)
        layer = sdf.Layer.FindOrOpen(usd)
        if layer is None:
            report.setdefault("unopened", []).append(usd)
            return usd
        dependencies_of = (
            getattr(layer, "GetCompositionAssetDependencies", None)
            or layer.GetExternalReferences
        )
        dependencies = [str(path) for path in dependencies_of()]
        child_copies: dict[str, str] = {}
        if depth_left > 0:
            for dependency in dependencies:
                if UNRESOLVABLE_ASSET_HOST in dependency or not _is_layer_path(
                    dependency
                ):
                    continue
                absolute = _anchor_asset_path(dependency, usd)
                prepared_child = self._prepare_layer(
                    sdf,
                    usd_utils,
                    absolute,
                    report,
                    prepared_by_source,
                    depth_left - 1,
                )
                if prepared_child != absolute:
                    child_copies[dependency] = prepared_child
        if not child_copies and not any(
            UNRESOLVABLE_ASSET_HOST in dependency for dependency in dependencies
        ):
            return usd
        assets_root = self._isaac.get_assets_root_path()
        if not assets_root:
            report["reason"] = "no assets root"
            return usd
        copy = sdf.Layer.CreateAnonymous(f"{stem}-prepared.usd")
        copy.TransferContent(layer)

        def rewrite(path: Any) -> str:
            raw = str(path)
            if raw in child_copies:
                report.setdefault("relinked", []).append((raw, child_copies[raw]))
                return child_copies[raw]
            return _prepared_asset_path(raw, usd, assets_root, self._usd_exists, report)

        usd_utils.ModifyAssetPaths(copy, rewrite)
        local.parent.mkdir(parents=True, exist_ok=True)
        if not copy.Export(str(local)):
            report["reason"] = f"could not write {local}"
            return usd
        report["reason"] = "prepared"
        prepared_by_source[usd] = str(local)
        return str(local)

    def create_camera(self, name: str, attrs: dict[str, Any]) -> "CameraHandle":
        self._require_booted()
        existing = self._handles.get(name)
        if existing is not None:
            old_attrs, handle = existing
            changed = {
                key
                for key in set(old_attrs) | set(attrs)
                if key not in self._RUNTIME_KEYS
                and old_attrs.get(key) != attrs.get(key)
            }
            if not changed:
                return handle
            restart_required = changed - self._CAMERA_MUTABLE_KEYS
            world_orientation_keys = {
                "target",
                "orientation_wxyz",
                "orientation_rpy_deg",
            }
            removed_orientation = {
                key
                for key in world_orientation_keys
                if key in old_attrs and key not in attrs
            }
            if (
                not attrs.get("parent_prim")
                and removed_orientation
                and not any(
                    attrs.get(key) is not None for key in world_orientation_keys
                )
            ):
                restart_required.update(removed_orientation)
            if (
                "position" in old_attrs
                and "position" not in attrs
                and not attrs.get("parent_prim")
                and not any(
                    attrs.get(key) is not None for key in world_orientation_keys
                )
            ):
                restart_required.add("position")
            if (
                attrs.get("prim_path")
                and "fov_deg" in old_attrs
                and "fov_deg" not in attrs
            ):
                restart_required.add("fov_deg")
            if restart_required:
                fields = ", ".join(sorted(restart_required))
                raise ValueError(
                    f"camera {name!r} requires a module restart to change: {fields}"
                )
            if not isinstance(handle, CameraHandle):
                raise TypeError(f"cached resource {name!r} is not a camera")

            def apply_update() -> None:
                if self._episode_id is not None:
                    raise RuntimeError(
                        f"camera {name!r} cannot be reconfigured during episode capture"
                    )
                handle.reconfigure(attrs)

            self.run(apply_update, timeout=120.0)
            self._handles[name] = (dict(attrs), handle)
            return handle
        if self.mock:
            factory = lambda: MockCameraHandle(name, attrs)
        else:
            factory = lambda: self.run(
                lambda: self._create_camera_isaac(name, attrs), timeout=120.0
            )
        return self._cached_handle("camera", name, attrs, factory)

    def _create_camera_isaac(
        self, name: str, attrs: dict[str, Any]
    ) -> "IsaacCameraHandle":
        from .spatial import to_vec3

        parent = attrs.get("parent_prim")
        if parent:
            self._require_prim(parent)
            prim_path = f"{parent.rstrip('/')}/{_prim_name(name)}"
        else:
            prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        width = int(attrs.get("width", 640))
        height = int(attrs.get("height", 480))

        kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            resolution=(width, height),
        )
        if attrs.get("position") is not None:
            kwargs["position"] = list(to_vec3(attrs.get("position")))

        cam = self._isaac.Camera(**kwargs)
        cam.initialize()
        _apply_camera_configuration(cam, attrs)
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

    def create_base(self, name: str, attrs: dict[str, Any]) -> "BaseHandle":
        self._require_booted()
        if self.mock:
            factory = lambda: MockBaseHandle(name, attrs)
        else:
            factory = lambda: self.run(
                lambda: self._create_base_isaac(name, attrs), timeout=120.0
            )
        return self._cached_handle("base", name, attrs, factory)

    def _create_base_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacBaseHandle":
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

        base_kwargs: dict[str, Any] = dict(
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
        from isaacsim.core.api.sensors import RigidContactView
    except ImportError:
        from omni.isaac.core.sensors import RigidContactView
    ns.RigidContactView = RigidContactView

    try:
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    except ImportError:
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
    ns.DynamicCuboid = DynamicCuboid
    ns.FixedCuboid = FixedCuboid

    try:
        from isaacsim.core.api.materials import PhysicsMaterial
    except ImportError:
        from omni.isaac.core.materials import PhysicsMaterial
    ns.PhysicsMaterial = PhysicsMaterial

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
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    except ImportError:
        from omni.isaac.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from omni.isaac.wheeled_robots.robots import WheeledRobot
    ns.WheeledRobot = WheeledRobot
    ns.DifferentialController = DifferentialController

    return ns


# ======================================================================
# Handles - the interface component models talk to. All public methods are
# safe to call from any thread.
# ======================================================================


DEFAULT_ARM_IS_MOVING_VELOCITY_THRESHOLD_RAD_S = 1e-2


@dataclass(frozen=True)
class JointTrajectoryPoint:
    time_s: float
    positions: tuple[float, ...]


class ArmHandle:
    _sim: "SimManager"

    def get_joint_positions(self) -> list[float]:  # radians
        raise NotImplementedError

    def _positions_on_sim_thread(self) -> list[float]:
        """Read joint positions (radians) while already on the sim thread."""
        raise NotImplementedError

    def joint_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def set_joint_targets(self, positions: list[float]) -> None:
        raise NotImplementedError

    def wait_for_joint_targets(
        self, targets: Sequence[float], tolerance_rad: float
    ) -> Future[None]:
        """Resolve once every joint is within tolerance of its target.

        The check runs on the sim thread once per simulator frame, so the
        caller waits on simulator progress instead of polling over RPC with
        wall-clock sleeps. Cancelling the future removes the check."""
        if len(targets) == 0:
            raise ValueError("wait_for_joint_targets needs at least one target")
        if not math.isfinite(tolerance_rad) or tolerance_rad < 0.0:
            raise ValueError("tolerance_rad must be finite and non-negative")
        future: Future[None] = Future()
        callback_name = f"arm_wait_{id(self)}_{id(future)}"
        wanted = tuple(float(value) for value in targets)

        def _check(_step_size: float) -> None:
            if future.done():
                self._sim._step_callbacks.pop(callback_name, None)
                return
            try:
                current = self._positions_on_sim_thread()
                arrived = len(current) >= len(wanted) and all(
                    abs(actual - target) <= tolerance_rad
                    for actual, target in zip(current, wanted)
                )
            except BaseException as exc:
                self._sim._step_callbacks.pop(callback_name, None)
                future.set_exception(exc)
                return
            if arrived:
                self._sim._step_callbacks.pop(callback_name, None)
                future.set_result(None)

        def _attach() -> None:
            self._sim._step_callbacks[callback_name] = _check
            _check(0.0)

        self._sim.run(_attach)
        return future

    def start_trajectory(self, points: Sequence[JointTrajectoryPoint]) -> Future[None]:
        """Apply timestamped targets from the simulator physics callback."""
        raise NotImplementedError

    def set_nudge(self, offsets: list[float]) -> None:
        """Drive toward the commanded targets plus offsets (radians)."""
        raise NotImplementedError

    def nudge_offsets(self) -> list[float] | None:
        return None

    def stop_trajectory(self) -> None:
        """Remove the trajectory callback and hold the current joint positions."""
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def dynamics_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_end_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """((x,y,z) meters, (w,x,y,z) quaternion) of the end effector."""
        raise NotImplementedError

    def capture_state(self):
        """Return positions, velocities, target, and EE pose on the sim thread."""
        raise NotImplementedError


class IsaacArmHandle(ArmHandle):
    def __init__(
        self,
        sim: SimManager,
        articulation,
        ee_prim,
        prim_path: str,
        joint_names: Sequence[str] | None = None,
        *,
        is_moving_velocity_threshold_rad_s: float = (
            DEFAULT_ARM_IS_MOVING_VELOCITY_THRESHOLD_RAD_S
        ),
        max_joint_deg_s: float | None = None,
        max_joint_deg_s2: float | None = None,
    ) -> None:
        if (
            not math.isfinite(is_moving_velocity_threshold_rad_s)
            or is_moving_velocity_threshold_rad_s < 0.0
        ):
            raise ValueError(
                "is_moving_velocity_threshold_rad_s must be finite and non-negative"
            )
        if max_joint_deg_s is not None and (
            not math.isfinite(max_joint_deg_s) or max_joint_deg_s <= 0.0
        ):
            raise ValueError("max_joint_deg_s must be finite and positive")
        if max_joint_deg_s2 is not None and (
            not math.isfinite(max_joint_deg_s2) or max_joint_deg_s2 <= 0.0
        ):
            raise ValueError("max_joint_deg_s2 must be finite and positive")
        self._sim = sim
        self._art = articulation
        self._ee = ee_prim
        self._prim_path = prim_path
        self._joint_names = tuple(joint_names or tuple(articulation.dof_names))
        self._joint_indices: list[int] = []
        self._is_moving_velocity_threshold_rad_s = is_moving_velocity_threshold_rad_s
        self._max_joint_rad_s = (
            math.radians(max_joint_deg_s) if max_joint_deg_s is not None else None
        )
        self._max_joint_acceleration_rad_s2 = (
            math.radians(max_joint_deg_s2) if max_joint_deg_s2 is not None else None
        )
        self.refresh_dofs()
        self._target: list[float] | None = None
        self._nudge: list[float] | None = None
        self._trajectory_callback_name = f"arm_trajectory_{id(self)}"
        self._trajectory: list[tuple[float, Any, tuple[float, ...]]] = []
        self._trajectory_index = 0
        self._trajectory_elapsed_s = 0.0
        self._trajectory_future: Future[None] | None = None
        self._trajectory_attached = False

    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def refresh_dofs(self) -> None:
        import numpy as np

        names = list(self._art.dof_names)
        missing = [name for name in self._joint_names if name not in names]
        if missing:
            raise ValueError(
                f"arm joints {missing} not found in articulation dof_names: {names}"
            )
        self._joint_indices = [names.index(name) for name in self._joint_names]
        if self._max_joint_rad_s is not None:
            self._art._articulation_view.set_max_joint_velocities(
                np.full((1, len(self._joint_indices)), self._max_joint_rad_s),
                joint_indices=np.asarray(self._joint_indices),
            )

    def replace_articulation(self, articulation) -> None:
        self._art = articulation

    def _positions_on_sim_thread(self) -> list[float]:
        return [
            float(v)
            for v in self._art.get_joint_positions(joint_indices=self._joint_indices)
        ]

    def get_joint_positions(self) -> list[float]:
        return self._sim.run(self._positions_on_sim_thread)

    def set_joint_targets(self, positions: list[float]) -> None:
        if len(positions) != len(self._joint_indices):
            raise ValueError(
                f"expected {len(self._joint_indices)} joint positions, got {len(positions)}"
            )

        def _apply():
            self._target = list(positions)
            self._apply_commanded()

        self._sim.run(_apply)

    def _apply_commanded(self, action: Any = None) -> None:
        """Drive toward the clean target plus any active nudge offset."""
        import numpy as np

        assert self._target is not None
        if self._nudge is None and action is not None:
            self._art.apply_action(action)
            return
        commanded = self._target
        if self._nudge is not None:
            commanded = [t + n for t, n in zip(self._target, self._nudge)]
        self._art.apply_action(
            self._sim._isaac.ArticulationAction(
                joint_positions=np.asarray(commanded, dtype=float),
                joint_indices=self._joint_indices,
            )
        )

    def set_nudge(self, offsets: list[float]) -> None:
        if len(offsets) != len(self._joint_indices):
            raise ValueError(
                f"expected {len(self._joint_indices)} nudge offsets, got {len(offsets)}"
            )
        self._nudge = list(offsets) if any(offsets) else None
        if self._target is not None:
            self._apply_commanded()

    def nudge_offsets(self) -> list[float] | None:
        return None if self._nudge is None else list(self._nudge)

    def start_trajectory(self, points: Sequence[JointTrajectoryPoint]) -> Future[None]:
        import numpy as np

        if not points:
            raise ValueError("trajectory must contain at least one point")
        future: Future[None] = Future()

        def _start() -> None:
            if self._trajectory_attached:
                raise RuntimeError("another arm trajectory is already active")
            trajectory: list[tuple[float, Any, tuple[float, ...]]] = []
            previous_time = -1.0
            for point in points:
                if not math.isfinite(point.time_s) or point.time_s < 0.0:
                    raise ValueError(
                        "trajectory point times must be finite and non-negative"
                    )
                if point.time_s <= previous_time:
                    raise ValueError("trajectory point times must strictly increase")
                if len(point.positions) != len(self._joint_indices):
                    raise ValueError(
                        f"expected {len(self._joint_indices)} joint positions, "
                        f"got {len(point.positions)}"
                    )
                if not all(math.isfinite(value) for value in point.positions):
                    raise ValueError("trajectory joint positions must be finite")
                positions = tuple(point.positions)
                action = self._sim._isaac.ArticulationAction(
                    joint_positions=np.asarray(positions, dtype=float),
                    joint_indices=self._joint_indices,
                )
                trajectory.append((point.time_s, action, positions))
                previous_time = point.time_s

            self._trajectory = trajectory
            self._trajectory_index = 0
            self._trajectory_elapsed_s = 0.0
            self._trajectory_future = future
            self._sim.world.add_physics_callback(
                self._trajectory_callback_name, self._on_trajectory_step
            )
            self._trajectory_attached = True

        self._sim.run(_start)
        return future

    def _on_trajectory_step(self, step_size: float) -> None:
        future = self._trajectory_future
        if future is None or future.done():
            return
        try:
            while (
                self._trajectory_index < len(self._trajectory)
                and self._trajectory[self._trajectory_index][0]
                <= self._trajectory_elapsed_s + 1e-9
            ):
                _, action, positions = self._trajectory[self._trajectory_index]
                self._target = list(positions)
                self._apply_commanded(action)
                self._trajectory_index += 1
            if self._trajectory_index == len(self._trajectory):
                future.set_result(None)
            else:
                self._trajectory_elapsed_s += float(step_size)
        except BaseException as exc:
            future.set_exception(exc)

    def stop_trajectory(self) -> None:
        def _stop() -> None:
            import numpy as np

            if self._trajectory_attached:
                self._sim.world.remove_physics_callback(self._trajectory_callback_name)
                self._trajectory_attached = False
            current = tuple(
                float(value)
                for value in self._art.get_joint_positions(
                    joint_indices=self._joint_indices
                )
            )
            velocities_before_raw = self._art.get_joint_velocities(
                joint_indices=self._joint_indices
            )
            velocities_before = (
                None
                if velocities_before_raw is None
                else tuple(float(value) for value in velocities_before_raw)
            )
            action = self._sim._isaac.ArticulationAction(
                joint_positions=np.asarray(current, dtype=float),
                joint_indices=self._joint_indices,
            )
            self._art.apply_action(action)
            self._art.set_joint_velocities(
                np.zeros(len(self._joint_indices), dtype=float),
                joint_indices=self._joint_indices,
            )
            velocities_after_raw = self._art.get_joint_velocities(
                joint_indices=self._joint_indices
            )
            velocities_after = (
                None
                if velocities_after_raw is None
                else tuple(float(value) for value in velocities_after_raw)
            )
            LOGGER.info(
                "arm %s stop command positions_rad=%s velocities_before_rad_s=%s "
                "velocities_after_clear_rad_s=%s previous_target_rad=%s",
                self._prim_path,
                current,
                velocities_before,
                velocities_after,
                self._target,
            )

            trace_name = f"arm_stop_trace_{id(self)}"
            sample_index = 0

            def _trace_stop(step_size: float) -> None:
                nonlocal sample_index
                try:
                    positions = tuple(
                        float(value)
                        for value in self._art.get_joint_positions(
                            joint_indices=self._joint_indices
                        )
                    )
                    velocities_raw = self._art.get_joint_velocities(
                        joint_indices=self._joint_indices
                    )
                    velocities = (
                        None
                        if velocities_raw is None
                        else tuple(float(value) for value in velocities_raw)
                    )
                    LOGGER.info(
                        "arm %s stop trace sample=%s step_size_s=%s "
                        "positions_rad=%s velocities_rad_s=%s hold_target_rad=%s",
                        self._prim_path,
                        sample_index,
                        step_size,
                        positions,
                        velocities,
                        self._target,
                    )
                    sample_index += 1
                    if sample_index >= 20:
                        self._sim._step_callbacks.pop(trace_name, None)
                except BaseException:
                    self._sim._step_callbacks.pop(trace_name, None)
                    LOGGER.exception("arm %s stop trace failed", self._prim_path)

            self._sim._step_callbacks[trace_name] = _trace_stop
            # The measured pose includes any active nudge; keep the target clean.
            nudge = self._nudge or [0.0] * len(current)
            self._target = [value - offset for value, offset in zip(current, nudge)]
            if self._trajectory_future is not None:
                self._trajectory_future.cancel()
            self._trajectory = []
            self._trajectory_future = None

        self._sim.run(_stop)

    def is_moving(self) -> bool:
        def _check():
            vels = self._art.get_joint_velocities(joint_indices=self._joint_indices)
            if vels is None:
                return False
            return bool(
                max(abs(float(v)) for v in vels)
                > self._is_moving_velocity_threshold_rad_s
            )

        return self._sim.run(_check)

    def stop(self) -> None:
        self.stop_trajectory()

    def dynamics_metadata(self) -> dict[str, Any]:
        def _metadata() -> dict[str, Any]:
            import numpy as np

            articulation = self._art
            controller = getattr(articulation, "_articulation_view", articulation)

            def values(method_names: tuple[str, ...]) -> list[float] | None:
                for method_name in method_names:
                    method = getattr(controller, method_name, None)
                    if callable(method):
                        raw: Any = method()
                        if raw is None:
                            return None
                        flattened = np.asarray(raw)
                        if flattened.ndim > 1:
                            flattened = flattened[0]
                        return [
                            float(flattened[index]) for index in self._joint_indices
                        ]
                return None

            result: dict[str, Any] = {
                "source": "isaac-articulation-controller",
                "prim_path": self._prim_path,
                "joint_names": list(self._joint_names),
                "controller_type": type(controller).__name__,
            }
            for key, method_names in (
                (
                    "max_velocity_rad_s",
                    ("get_joint_max_velocities", "get_max_velocities"),
                ),
                ("max_effort", ("get_max_efforts",)),
            ):
                raw_values = values(method_names)
                if raw_values is not None:
                    result[key] = raw_values
            if self._max_joint_acceleration_rad_s2 is not None:
                result["max_acceleration_rad_s2"] = [
                    self._max_joint_acceleration_rad_s2
                ] * len(self._joint_indices)
            get_gains = getattr(controller, "get_gains", None)
            if callable(get_gains):
                gains: Any = get_gains()
                stiffness = np.asarray(gains[0])
                damping = np.asarray(gains[1])
                if stiffness.ndim > 1:
                    stiffness = stiffness[0]
                    damping = damping[0]
                result["drive_stiffness"] = [
                    float(stiffness[index]) for index in self._joint_indices
                ]
                result["drive_damping"] = [
                    float(damping[index]) for index in self._joint_indices
                ]
            return result

        return self._sim.run(_metadata)

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

    def capture_state(self):
        positions = [
            float(v)
            for v in self._art.get_joint_positions(joint_indices=self._joint_indices)
        ]
        raw_velocities = self._art.get_joint_velocities(
            joint_indices=self._joint_indices
        )
        velocities = (
            [0.0] * len(positions)
            if raw_velocities is None
            else [float(v) for v in raw_velocities]
        )
        ee_position, ee_rotation = self.get_end_pose()
        return (
            positions,
            velocities,
            positions if self._target is None else list(self._target),
            ee_position,
            ee_rotation,
        )


class MockArmHandle(ArmHandle):
    """Joints move linearly toward their targets at a fixed speed."""

    SPEED = 1.0  # rad/s per joint

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        dof = int(attrs.get("mock_dof", 6))
        meta = KNOWN_ASSETS.get(str(attrs.get("asset", "")), {})
        configured_names = tuple(meta.get("joint_names", ()))
        self._joint_names = (
            configured_names
            if len(configured_names) == dof
            else tuple(f"joint{index + 1}" for index in range(dof))
        )
        raw_acceleration_limit = attrs.get("max_joint_deg_s2")
        self._max_joint_acceleration_rad_s2 = (
            math.radians(float(raw_acceleration_limit))
            if raw_acceleration_limit is not None
            else None
        )
        self._lock = threading.Lock()
        self._start = [0.0] * dof
        self._target = [0.0] * dof
        self._t0 = time.monotonic()
        self._trajectory_callback_name = f"arm_trajectory_{id(self)}"
        self._trajectory: tuple[JointTrajectoryPoint, ...] = ()
        self._trajectory_index = 0
        self._trajectory_elapsed_s = 0.0
        self._trajectory_future: Future[None] | None = None
        self._trajectory_attached = False

    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def _positions_at(self, now: float) -> list[float]:
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

    def get_joint_positions(self) -> list[float]:
        with self._lock:
            return self._positions_at(time.monotonic())

    def _positions_on_sim_thread(self) -> list[float]:
        return self.get_joint_positions()

    def set_joint_targets(self, positions: list[float]) -> None:
        with self._lock:
            now = time.monotonic()
            self._start = self._positions_at(now)
            if len(positions) != len(self._start):
                raise ValueError(
                    f"expected {len(self._start)} joint positions, got {len(positions)}"
                )
            self._target = list(positions)
            self._t0 = now

    def start_trajectory(self, points: Sequence[JointTrajectoryPoint]) -> Future[None]:
        if not points:
            raise ValueError("trajectory must contain at least one point")
        trajectory = tuple(points)
        previous_time = -1.0
        for point in trajectory:
            if not math.isfinite(point.time_s) or point.time_s < 0.0:
                raise ValueError(
                    "trajectory point times must be finite and non-negative"
                )
            if point.time_s <= previous_time:
                raise ValueError("trajectory point times must strictly increase")
            if len(point.positions) != len(self._joint_names):
                raise ValueError(
                    f"expected {len(self._joint_names)} joint positions, "
                    f"got {len(point.positions)}"
                )
            if not all(math.isfinite(value) for value in point.positions):
                raise ValueError("trajectory joint positions must be finite")
            previous_time = point.time_s

        future: Future[None] = Future()

        def _start() -> None:
            if self._trajectory_attached:
                raise RuntimeError("another arm trajectory is already active")
            self._trajectory = trajectory
            self._trajectory_index = 0
            self._trajectory_elapsed_s = 0.0
            self._trajectory_future = future
            self._sim._step_callbacks[self._trajectory_callback_name] = (
                self._on_trajectory_step
            )
            self._trajectory_attached = True

        self._sim.run(_start)
        return future

    def _on_trajectory_step(self, step_size: float) -> None:
        future = self._trajectory_future
        if future is None or future.done():
            return
        try:
            while (
                self._trajectory_index < len(self._trajectory)
                and self._trajectory[self._trajectory_index].time_s
                <= self._trajectory_elapsed_s + 1e-9
            ):
                self.set_joint_targets(
                    list(self._trajectory[self._trajectory_index].positions)
                )
                self._trajectory_index += 1
            if self._trajectory_index == len(self._trajectory):
                future.set_result(None)
            else:
                self._trajectory_elapsed_s += float(step_size)
        except BaseException as exc:
            future.set_exception(exc)

    def stop_trajectory(self) -> None:
        def _stop() -> None:
            self._sim._step_callbacks.pop(self._trajectory_callback_name, None)
            self._trajectory_attached = False
            with self._lock:
                now = time.monotonic()
                self._start = self._positions_at(now)
                self._target = list(self._start)
                self._t0 = now
            if self._trajectory_future is not None:
                self._trajectory_future.cancel()
            self._trajectory = ()
            self._trajectory_future = None

        self._sim.run(_stop)

    def is_moving(self) -> bool:
        with self._lock:
            pos = self._positions_at(time.monotonic())
            return any(abs(p - t) > 1e-9 for p, t in zip(pos, self._target))

    def stop(self) -> None:
        self.stop_trajectory()

    def dynamics_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source": "mock-linear-arm",
            "joint_names": list(self._joint_names),
            "max_velocity_rad_s": [self.SPEED] * len(self._joint_names),
        }
        if self._max_joint_acceleration_rad_s2 is not None:
            result["max_acceleration_rad_s2"] = [
                self._max_joint_acceleration_rad_s2
            ] * len(self._joint_names)
        return result

    def get_end_pose(self):
        # a fixed, deterministic pose for testing
        return ((0.3, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0))

    def capture_state(self):
        with self._lock:
            positions = self._positions_at(time.monotonic())
            velocities = [
                (
                    0.0
                    if math.isclose(position, target)
                    else math.copysign(self.SPEED, target - position)
                )
                for position, target in zip(positions, self._target)
            ]
            return (
                positions,
                velocities,
                list(self._target),
                (0.3, 0.0, 0.3),
                (1.0, 0.0, 0.0, 0.0),
            )


GRIPPER_OPEN_WIDTH_M = 0.085
GRIPPER_HOLDING_SETTLE_S = 0.25
GRIPPER_DRIVE_STIFFNESS = 30.0
GRIPPER_DRIVE_DAMPING = 1.0
GRIPPER_COUPLED_JOINTS = (
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
)


class GripperHandle:
    def jaw_limits(self) -> tuple[float, float]:
        """Return raw actuator positions as (fully open, fully closed)."""
        raise NotImplementedError

    def jaw_unit(self) -> str:
        raise NotImplementedError

    def get_jaw(self) -> float:
        raise NotImplementedError

    def set_jaw(self, position: float) -> None:
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError

    def is_holding(self) -> bool:
        raise NotImplementedError

    def capture_state(self) -> tuple[float, float]:
        """Return policy command and observation state in the gripper's mode."""
        raise NotImplementedError


class IsaacGripperHandle(GripperHandle):
    def __init__(
        self,
        sim: SimManager,
        articulation,
        drive_joint: str,
        open_position: float,
        closed_position: float,
        holding_tolerance: float,
        *,
        binary: bool = False,
        position_unit: str = "rad",
        minimum_grasp_closure: float = 0.0,
        drive_stiffness: float = GRIPPER_DRIVE_STIFFNESS,
        drive_damping: float = GRIPPER_DRIVE_DAMPING,
        coupled_drive_joints: Sequence[str] = (),
        contact_view: Any = None,
        contact_dt: float = 1.0 / 60.0,
    ) -> None:
        if math.isclose(open_position, closed_position):
            raise ValueError("gripper open and closed positions must differ")
        self._sim = sim
        self._art = articulation
        self._open = open_position
        self._closed = closed_position
        self._holding_tolerance = holding_tolerance
        self._binary = binary
        self._position_unit = position_unit
        self._minimum_grasp_closure = minimum_grasp_closure
        self._drive_stiffness = drive_stiffness
        self._drive_damping = drive_damping
        self._contact_view = contact_view
        self._contact_dt = contact_dt
        self._last_contact_forces = (0.0, 0.0)
        self._max_contact_forces = (0.0, 0.0)
        self._progress_epsilon = max(
            abs(closed_position - open_position) / 100.0,
            holding_tolerance / 2.0,
        )
        names = list(articulation.dof_names)
        self._index = names.index(drive_joint)
        self._drive_indices = [self._index]
        self._measure_indices = [self._index] + [
            names.index(name) for name in coupled_drive_joints if name in names
        ]
        self._target: float | None = None
        self._has_progressed = False
        self._best_gap: float | None = None
        self._last_progress_sim_time: float | None = None
        self._held_latch = False
        self._effort_closing = False
        self._configure_drives(names)
        if contact_view is not None:
            # The close effort must be re-applied every physics substep, not
            # once per rendered frame, so hold strength does not depend on the
            # rendering cadence.
            self._sim.world.add_physics_callback(
                f"gripper_hold_{id(self)}", self._on_physics_step
            )

    def _configure_drives(self, names: Sequence[str]) -> None:
        indices = [
            names.index(name) for name in GRIPPER_COUPLED_JOINTS if name in names
        ]
        try:
            import numpy as np

            controller = self._art.get_articulation_controller()
            kps, kds = controller.get_gains()
            kps = np.array(kps, dtype=float).copy()
            kds = np.array(kds, dtype=float).copy()
            for index in indices:
                kps[index] = 0.0
                kds[index] = 0.1
            for index in self._drive_indices:
                kps[index] = self._drive_stiffness
                kds[index] = self._drive_damping
            controller.set_gains(kps=kps, kds=kds)
        except Exception as exc:
            raise RuntimeError("could not configure gripper joint drives") from exc

    def _set_drive_gains(self, stiffness: float, damping: float) -> None:
        import numpy as np

        controller = self._art.get_articulation_controller()
        kps, kds = controller.get_gains()
        kps = np.array(kps, dtype=float).copy()
        kds = np.array(kds, dtype=float).copy()
        for index in self._drive_indices:
            kps[index] = stiffness
            kds[index] = damping
        controller.set_gains(kps=kps, kds=kds)

    def jaw_limits(self) -> tuple[float, float]:
        return self._open, self._closed

    def jaw_unit(self) -> str:
        return self._position_unit

    def _mean_jaw_position(self) -> float:
        positions = [
            float(value)
            for value in self._art.get_joint_positions(
                joint_indices=self._measure_indices
            )
        ]
        return sum(positions) / len(positions)

    def get_jaw(self) -> float:
        return self._sim.run(self._mean_jaw_position)

    def _contact_forces(self) -> tuple[float, float] | None:
        contact_view = getattr(self, "_contact_view", None)
        if contact_view is None:
            return None
        import numpy as np

        forces = contact_view.get_contact_force_matrix(dt=self._contact_dt)
        if forces is None:
            return None
        if hasattr(forces, "detach"):
            forces = forces.detach().cpu().numpy()
        elif hasattr(forces, "numpy"):
            forces = forces.numpy()
        values = np.asarray(forces, dtype=float)
        if values.shape[0] != 2:
            raise RuntimeError(
                f"expected two finger contact sensors, got {values.shape}"
            )
        magnitudes = np.linalg.norm(values, axis=-1).sum(axis=1)
        current = float(magnitudes[0]), float(magnitudes[1])
        self._last_contact_forces = current
        self._max_contact_forces = tuple(
            max(previous, value)
            for previous, value in zip(self._max_contact_forces, current)
        )
        return current

    def contact_force_summary(self) -> dict[str, list[float]] | None:
        if self._contact_view is None:
            return None

        def read() -> dict[str, list[float]]:
            current = self._contact_forces() or self._last_contact_forces
            return {
                "contact_force_n": list(current),
                "max_contact_force_n": list(self._max_contact_forces),
            }

        return self._sim.run(read)

    def _apply_close_effort(self) -> None:
        import numpy as np

        direction = 1.0 if self._closed > self._open else -1.0
        self._art.apply_action(
            self._sim._isaac.ArticulationAction(
                joint_efforts=np.full(
                    len(self._drive_indices), direction * GRIPPER_HOLD_FORCE_N
                ),
                joint_indices=self._drive_indices,
            )
        )


    def _start_effort_close(self) -> None:
        def apply() -> None:
            initial_gap = abs(self._closed - self._mean_jaw_position())
            self._set_drive_gains(0.0, 0.0)
            self._target = self._closed
            self._has_progressed = False
            self._best_gap = initial_gap
            self._last_progress_sim_time = self._sim_time()
            self._held_latch = False
            self._last_contact_forces = (0.0, 0.0)
            self._max_contact_forces = (0.0, 0.0)
            self._effort_closing = True
            self._apply_close_effort()

        self._sim.run(apply)

    def _on_physics_step(self, _step_size: float) -> None:
        if self._effort_closing:
            self._apply_close_effort()
            if not self._held_latch:
                self._gap_and_stall()

    def set_jaw(self, position: float) -> None:
        import numpy as np

        low, high = sorted((self._open, self._closed))
        target = min(max(float(position), low), high)

        def apply() -> None:
            initial_gap = abs(target - self._mean_jaw_position())
            self._effort_closing = False
            self._set_drive_gains(self._drive_stiffness, self._drive_damping)
            self._art.apply_action(
                self._sim._isaac.ArticulationAction(
                    joint_positions=np.full(
                        len(self._drive_indices), target, dtype=float
                    ),
                    joint_efforts=np.zeros(len(self._drive_indices), dtype=float),
                    joint_indices=self._drive_indices,
                )
            )
            self._target = target
            self._has_progressed = False
            self._best_gap = initial_gap
            self._last_progress_sim_time = self._sim_time()
            self._held_latch = False
            self._last_contact_forces = (0.0, 0.0)
            self._max_contact_forces = (0.0, 0.0)

        self._sim.run(apply)

    def open(self) -> None:
        self.set_jaw(self._open)

    def close(self) -> None:
        if self._contact_view is None:
            self.set_jaw(self._closed)
        else:
            self._start_effort_close()

    def stop(self) -> None:
        self.set_jaw(self.get_jaw())

    def _sim_time(self) -> float:
        if self._sim.mock:
            return self._sim._mock_sim_time_s
        return float(self._sim.world.current_time)

    def _gap_and_stall(self) -> tuple[float, bool]:
        if self._target is None:
            return 0.0, False
        measured = self._mean_jaw_position()
        gap = abs(self._target - measured)
        if self._best_gap is None or gap < self._best_gap - self._progress_epsilon:
            if self._best_gap is not None:
                self._has_progressed = True
            self._best_gap = gap
            self._last_progress_sim_time = self._sim_time()
        closure = abs(measured - self._open) / abs(self._closed - self._open)
        contact_view = getattr(self, "_contact_view", None)
        position_stalled = (
            contact_view is None
            and math.isclose(self._target, self._closed)
            and self._has_progressed
            and self._last_progress_sim_time is not None
            and self._sim_time() - self._last_progress_sim_time
            >= GRIPPER_HOLDING_SETTLE_S
            and gap > self._holding_tolerance
            and closure >= self._minimum_grasp_closure
        )
        force_stalled = False
        if contact_view is not None and math.isclose(self._target, self._closed):
            self._contact_forces()
            bilateral = all(
                force >= GRIPPER_CONTACT_FORCE_N for force in self._max_contact_forces
            )
            if bilateral and closure >= self._minimum_grasp_closure:
                force_stalled = True
        contact_settled = position_stalled or force_stalled
        if gap <= self._holding_tolerance:
            self._held_latch = False
        elif contact_settled:
            self._held_latch = True
        return gap, contact_settled

    def is_moving(self) -> bool:
        def check() -> bool:
            gap, stalled = self._gap_and_stall()
            return (
                gap > self._holding_tolerance and not stalled and not self._held_latch
            )

        return self._sim.run(check)

    def is_holding(self) -> bool:
        def check() -> bool:
            gap, _stalled = self._gap_and_stall()
            return gap > self._holding_tolerance and self._held_latch

        return self._sim.run(check)

    def capture_state(self) -> tuple[float, float]:
        measured = float(self._art.get_joint_positions(joint_indices=[self._index])[0])
        target = measured if self._target is None else self._target
        span = self._closed - self._open
        normalize = lambda value: min(max((value - self._open) / span, 0.0), 1.0)
        command = normalize(target)
        if self._binary:
            state = float(command >= 0.5)
            return state, state
        return command, normalize(measured)


class MockGripperHandle(GripperHandle):
    def __init__(self, name: str, attrs: dict[str, Any], arm: ArmHandle) -> None:
        self.name = name
        self._arm = arm
        meta = KNOWN_ASSETS[str(attrs.get("asset", "robotiq_2f_85"))]
        self._position_unit = str(meta.get("position_unit", "rad"))
        self._binary = str(meta.get("gripper_mode", "position")) == "binary"
        if self._position_unit == "rad":
            self._open = math.radians(float(attrs.get("open_deg", meta["open_deg"])))
            self._closed = math.radians(
                float(attrs.get("closed_deg", meta["closed_deg"]))
            )
            self._holding_tolerance = math.radians(
                float(attrs.get("holding_tolerance_deg", 2.0))
            )
        else:
            self._open = float(attrs.get("open_position", meta["open_position"]))
            self._closed = float(attrs.get("closed_position", meta["closed_position"]))
            self._holding_tolerance = float(
                attrs.get("holding_tolerance_m", meta["holding_tolerance"])
            )
        if math.isclose(self._open, self._closed):
            raise ValueError("gripper open and closed positions must differ")
        width = attrs.get("mock_object_width_m")
        self._object_width = None if width is None else float(width)
        self._open_width = float(meta.get("open_width_m", GRIPPER_OPEN_WIDTH_M))
        self._minimum_grasp_closure = float(meta.get("minimum_grasp_closure", 0.0))
        self._speed = abs(self._closed - self._open) * 2.0
        self._lock = threading.Lock()
        self._start = self._open
        self._target = self._open
        self._started = time.monotonic()

    def _contact_position(self) -> float:
        if self._object_width is None:
            return self._closed
        width = min(max(self._object_width, 0.0), self._open_width)
        fraction = 1.0 - width / self._open_width
        return self._open + fraction * (self._closed - self._open)

    def _effective_target(self) -> float:
        span = self._closed - self._open
        closure = (self._target - self._open) / span
        if closure <= 0.0:
            return self._target
        contact = (self._contact_position() - self._open) / span
        return self._open + min(closure, contact) * span

    def _jaw_at(self, now: float) -> float:
        target = self._effective_target()
        delta = target - self._start
        travel = min(self._speed * max(0.0, now - self._started), abs(delta))
        if travel >= abs(delta):
            return target
        return self._start + math.copysign(travel, delta)

    def jaw_limits(self) -> tuple[float, float]:
        return self._open, self._closed

    def jaw_unit(self) -> str:
        return self._position_unit

    def get_jaw(self) -> float:
        with self._lock:
            return self._jaw_at(time.monotonic())

    def set_jaw(self, position: float) -> None:
        low, high = sorted((self._open, self._closed))
        target = min(max(float(position), low), high)
        with self._lock:
            now = time.monotonic()
            self._start = self._jaw_at(now)
            self._target = target
            self._started = now

    def open(self) -> None:
        self.set_jaw(self._open)

    def close(self) -> None:
        self.set_jaw(self._closed)

    def stop(self) -> None:
        self.set_jaw(self.get_jaw())

    def is_moving(self) -> bool:
        with self._lock:
            return abs(self._jaw_at(time.monotonic()) - self._effective_target()) > 1e-9

    def is_holding(self) -> bool:
        with self._lock:
            measured = self._jaw_at(time.monotonic())
            at_contact = abs(measured - self._effective_target()) <= 1e-9
            span = self._closed - self._open
            closure = abs(measured - self._open) / abs(span)
            blocked = abs(self._target - measured) > self._holding_tolerance
            return at_contact and blocked and closure >= self._minimum_grasp_closure

    def capture_state(self) -> tuple[float, float]:
        with self._lock:
            measured = self._jaw_at(time.monotonic())
            span = self._closed - self._open
            normalize = lambda value: min(max((value - self._open) / span, 0.0), 1.0)
            command = normalize(self._target)
            if self._binary:
                state = float(command >= 0.5)
                return state, state
            return command, normalize(measured)


class CameraHandle:
    def get_rgb(self):
        """Return an (H, W, 3) uint8 numpy array."""
        raise NotImplementedError

    def cosmos_camera(self) -> tuple[str, tuple[int, int]]:
        """Return camera prim and resolution for a dedicated render product."""
        raise NotImplementedError

    def capture_rgb(self):
        """Copy RGB state while already on the sim thread."""
        raise NotImplementedError

    def reconfigure(self, attrs: dict[str, Any]) -> None:
        """Apply live-mutable camera configuration on the simulator thread."""
        raise NotImplementedError


class IsaacCameraHandle(CameraHandle):
    def __init__(self, sim: SimManager, camera) -> None:
        self._sim = sim
        self._cam = camera

    def reconfigure(self, attrs: dict[str, Any]) -> None:
        _apply_camera_configuration(self._cam, attrs)

    def capture_rgb(self):
        frame = self._cam.get_rgba()
        if frame is None or frame.size == 0:
            raise RuntimeError("no frame available yet - is the simulation playing?")
        return frame[:, :, :3].copy()

    def cosmos_camera(self) -> tuple[str, tuple[int, int]]:
        return str(self._cam.prim_path), tuple(self._cam.get_resolution())

    def get_rgb(self):
        return self._sim.run(self.capture_rgb)


class MockCameraHandle(CameraHandle):
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self._attrs = dict(attrs)
        self._w = int(attrs.get("width", 640))
        self._h = int(attrs.get("height", 480))

    def reconfigure(self, attrs: dict[str, Any]) -> None:
        self._attrs = dict(attrs)

    def get_rgb(self):
        import numpy as np

        # gradient background with a time-based moving bar so images change
        img = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, self._w, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(0, 255, self._h, dtype=np.uint8)[:, None]
        x = int((time.monotonic() * 60) % self._w)
        img[:, max(0, x - 5) : x + 5, :] = 255
        return img

    def capture_rgb(self):
        return self.get_rgb()


class BaseHandle:
    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError


class IsaacBaseHandle(BaseHandle):
    def __init__(
        self, sim: SimManager, robot, controller, wheel_radius: float, wheel_base: float
    ) -> None:
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
            self._robot.apply_wheel_actions(
                self._controller.forward(command=[lin, ang])
            )
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
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
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


UNRESOLVABLE_ASSET_HOST = "isaac-dev"
ASSETS_PATH_MARKER = "/Isaac/"
ASSET_CACHE_DIRNAME = "viam-isaac-sim-assets"
ASSET_LAYER_CACHE_VERSION = 1
ASSET_PREPARE_MAX_DEPTH = 4
PAD_PRIM_NAME_FRAGMENTS = ("pad", "fingertip", "inner_finger", "uflite_finger")


def _is_layer_path(asset_path: str) -> bool:
    if not (asset_path.startswith(("./", "../", "/")) or "://" in asset_path):
        return False
    return asset_path.rsplit("?", 1)[0].lower().endswith((".usd", ".usda", ".usdc"))


def asset_cache_dir() -> Path:
    base = os.environ.get("VIAM_MODULE_DATA") or tempfile.gettempdir()
    return Path(base) / ASSET_CACHE_DIRNAME


def _bucket_candidate(asset_path: str, assets_root: str) -> str | None:
    marker_at = asset_path.find(ASSETS_PATH_MARKER)
    if UNRESOLVABLE_ASSET_HOST not in asset_path or marker_at < 0:
        return None
    return assets_root.rstrip("/") + asset_path[marker_at:]


def _anchor_asset_path(asset_path: str, anchor_layer: str) -> str:
    if not asset_path.startswith(("./", "../")):
        return asset_path
    segments = anchor_layer.rsplit("/", 1)[0].split("/")
    floor = 3 if "://" in anchor_layer else 1
    for part in asset_path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if len(segments) > floor:
                segments.pop()
        else:
            segments.append(part)
    return "/".join(segments)


def _prepared_asset_path(
    asset_path: str,
    anchor_layer: str,
    assets_root: str,
    exists: Callable[[str], bool | None],
    report: dict[str, Any],
) -> str:
    candidate = _bucket_candidate(asset_path, assets_root)
    if candidate is not None:
        if exists(candidate) is False:
            report["missing"].append(candidate)
            return ""
        report["applied"].append((asset_path, candidate))
        return candidate
    anchored = _anchor_asset_path(asset_path, anchor_layer)
    if anchored != asset_path:
        report["anchored"].append((asset_path, anchored))
    return anchored


def _prim_range(usd: Any, root_prim: Any) -> Any:
    traverse_instances = getattr(usd, "TraverseInstanceProxies", None)
    if traverse_instances is None:
        return usd.PrimRange(root_prim)
    return usd.PrimRange(root_prim, traverse_instances())


def _remove_articulation_roots(
    usd: Any, usd_physics: Any, physx_schema: Any, root_prim: Any
) -> list[str]:
    removed: list[str] = []
    for prim in _prim_range(usd, root_prim):
        if prim.IsInstanceProxy() or not prim.HasAPI(usd_physics.ArticulationRootAPI):
            continue
        if prim.HasAPI(physx_schema.PhysxArticulationAPI):
            prim.RemoveAPI(physx_schema.PhysxArticulationAPI)
        prim.RemoveAPI(usd_physics.ArticulationRootAPI)
        removed.append(str(prim.GetPath()))
    return removed


def _pad_collision_status(
    usd: Any, usd_physics: Any, root_prim: Any
) -> list[tuple[str, bool, bool]]:
    status: list[tuple[str, bool, bool]] = []
    for prim in _prim_range(usd, root_prim):
        if not any(
            fragment in prim.GetName().lower() for fragment in PAD_PRIM_NAME_FRAGMENTS
        ):
            continue
        on_self = bool(prim.HasAPI(usd_physics.CollisionAPI))
        in_subtree = on_self or any(
            child.HasAPI(usd_physics.CollisionAPI) for child in _prim_range(usd, prim)
        )
        status.append((str(prim.GetPath()), on_self, bool(in_subtree)))
    return status


def _describe_composition(
    usd: Any, sdf: Any, root_prim: Any, usd_path: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "children": [],
        "prim_count": 0,
        "layer_default_prim": "",
        "layer_root_prims": [],
    }
    try:
        result["children"] = [child.GetName() for child in root_prim.GetChildren()]
        result["prim_count"] = sum(1 for _ in _prim_range(usd, root_prim))
    except Exception:
        LOGGER.exception("could not inspect gripper composition at %s", root_prim)
    try:
        layer = sdf.Layer.FindOrOpen(usd_path)
        if layer is not None:
            result["layer_default_prim"] = str(layer.defaultPrim)
            result["layer_root_prims"] = [str(spec.path) for spec in layer.rootPrims]
    except Exception:
        LOGGER.exception("could not inspect gripper asset layer %s", usd_path)
    return result
