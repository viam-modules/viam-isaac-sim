"""Versioned native episode contract shared by capture and dataset exporters."""
# ruff: noqa: FURB162

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 3
MINIMUM_STAGE_FRAMES = 33
JOINT_COUNT = 6
OUTCOMES = frozenset({"success", "failure", "incomplete"})
CAMERA_ROLES = frozenset({"fixed", "fixed_2", "wrist"})
GRIPPER_MODES = frozenset({"binary", "position"})


def _finite_vector(name: str, values: Sequence[float], size: int) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must have {size} values, got {len(values)}")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _unit_interval(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _quaternion(
    name: str, values: Sequence[float]
) -> tuple[float, float, float, float]:
    result = _finite_vector(name, values, 4)
    norm = math.sqrt(sum(value * value for value in result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(f"{name} must be a unit quaternion in (w, x, y, z) order")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class CameraStreamV2:
    role: str
    component_name: str
    file: str
    width: int
    height: int
    encoding: str = "h264"

    def __post_init__(self) -> None:
        if self.role not in CAMERA_ROLES:
            raise ValueError(f"camera role must be one of {sorted(CAMERA_ROLES)}")
        if not self.component_name:
            raise ValueError("camera component_name must not be empty")
        if not self.file.endswith(".mp4") or "/" in self.file or "\\" in self.file:
            raise ValueError("camera file must be a basename ending in .mp4")
        if isinstance(self.width, bool) or self.width <= 0:
            raise ValueError("camera width must be a positive integer")
        if isinstance(self.height, bool) or self.height <= 0:
            raise ValueError("camera height must be a positive integer")
        if self.encoding != "h264":
            raise ValueError("episode-v2 camera encoding must be h264")


@dataclass(frozen=True)
class RobotDescriptorV2:
    arm_component: str
    arm_model: str
    joint_names: tuple[str, ...]
    gripper_component: str
    gripper_model: str
    gripper_mode: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.arm_component,
                self.arm_model,
                self.gripper_component,
                self.gripper_model,
            )
        ):
            raise ValueError("robot component and model names must not be empty")
        if self.gripper_mode not in GRIPPER_MODES:
            raise ValueError(f"gripper_mode must be one of {sorted(GRIPPER_MODES)}")
        if len(self.joint_names) != JOINT_COUNT or any(
            not name for name in self.joint_names
        ):
            raise ValueError(
                f"robot joint_names must contain exactly {JOINT_COUNT} names"
            )
        if len(set(self.joint_names)) != JOINT_COUNT:
            raise ValueError("robot joint_names must be unique")


@dataclass(frozen=True)
class TargetPoseV2:
    prop_name: str
    position_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.prop_name:
            raise ValueError("target prop_name must not be empty")
        object.__setattr__(
            self, "position_m", _finite_vector("target position_m", self.position_m, 3)
        )
        object.__setattr__(
            self,
            "rotation_wxyz",
            _quaternion("target rotation_wxyz", self.rotation_wxyz),
        )


@dataclass(frozen=True)
class EpisodeSampleV2:
    frame_index: int
    sim_time_s: float
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_target_rad: tuple[float, ...]
    ee_position_m: tuple[float, float, float]
    ee_rotation_wxyz: tuple[float, float, float, float]
    gripper_command: float
    gripper_state: float
    joint_target_nudge_rad: tuple[float, ...] = (0.0,) * JOINT_COUNT

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        sim_time_s = float(self.sim_time_s)
        if not math.isfinite(sim_time_s) or sim_time_s < 0.0:
            raise ValueError("sim_time_s must be finite and non-negative")
        object.__setattr__(self, "sim_time_s", sim_time_s)
        for field_name in (
            "joint_position_rad",
            "joint_velocity_rad_s",
            "joint_target_rad",
            "joint_target_nudge_rad",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_vector(field_name, getattr(self, field_name), JOINT_COUNT),
            )
        object.__setattr__(
            self,
            "ee_position_m",
            _finite_vector("ee_position_m", self.ee_position_m, 3),
        )
        object.__setattr__(
            self,
            "ee_rotation_wxyz",
            _quaternion("ee_rotation_wxyz", self.ee_rotation_wxyz),
        )
        object.__setattr__(
            self,
            "gripper_command",
            _unit_interval("gripper_command", self.gripper_command),
        )
        object.__setattr__(
            self,
            "gripper_state",
            _unit_interval("gripper_state", self.gripper_state),
        )


@dataclass(frozen=True)
class EpisodeStageV3:
    start_frame: int
    end_frame_exclusive: int
    task: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_frame, bool)
            or not isinstance(self.start_frame, int)
            or self.start_frame < 0
        ):
            raise ValueError("stage start_frame must be a non-negative integer")
        if isinstance(self.end_frame_exclusive, bool) or not isinstance(
            self.end_frame_exclusive, int
        ):
            raise ValueError("stage end_frame_exclusive must be an integer")
        if self.end_frame_exclusive - self.start_frame < MINIMUM_STAGE_FRAMES:
            raise ValueError(
                f"stage must contain at least {MINIMUM_STAGE_FRAMES} frames"
            )
        if not self.task.strip():
            raise ValueError("stage task must not be empty")


@dataclass(frozen=True)
class EpisodeManifestV3:
    episode_id: str
    seed: int
    started_at: str
    robot: RobotDescriptorV2
    cameras: tuple[CameraStreamV2, ...]
    target: TargetPoseV2
    task: str
    stages: tuple[EpisodeStageV3, ...]
    outcome: str
    sample_count: int
    sample_rate_hz: float
    schema_version: int = SCHEMA_VERSION
    samples_file: str = "samples.parquet"

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not self.episode_id or "/" in self.episode_id or "\\" in self.episode_id:
            raise ValueError("episode_id must be a non-empty path-safe identifier")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        try:
            parsed = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("started_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("started_at must include a timezone")
        sample_rate_hz = float(self.sample_rate_hz)
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
        if isinstance(self.sample_count, bool) or self.sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if not self.task.strip():
            raise ValueError("task must not be empty")
        roles = [camera.role for camera in self.cameras]
        role_set = set(roles)
        if len(roles) != len(role_set) or not {"fixed", "wrist"} <= role_set:
            raise ValueError(
                "episode-v3 requires one fixed and one wrist camera, with an optional fixed_2 camera"
            )
        if role_set - CAMERA_ROLES:
            raise ValueError(
                f"unsupported camera roles: {sorted(role_set - CAMERA_ROLES)}"
            )
        if len({camera.file for camera in self.cameras}) != len(self.cameras):
            raise ValueError("camera stream filenames must be unique")
        if self.samples_file != "samples.parquet":
            raise ValueError("episode-v3 samples_file must be samples.parquet")
        if self.stages:
            if self.stages[0].start_frame != 0:
                raise ValueError("first stage must start at frame 0")
            for previous, current in zip(self.stages, self.stages[1:]):
                if previous.end_frame_exclusive != current.start_frame:
                    raise ValueError("stages must be contiguous")
            if self.stages[-1].end_frame_exclusive != self.sample_count:
                raise ValueError("final stage must end at sample_count")

    def to_dict(self) -> dict[str, Any]:
        """Return only JSON-native containers and scalar values."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
