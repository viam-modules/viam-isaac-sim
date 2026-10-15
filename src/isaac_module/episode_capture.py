"""Bounded asynchronous writer for synchronized native episode samples."""

from __future__ import annotations

import math
import os
import queue
import threading
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from .episode_schema import (
    CameraStreamV2,
    EpisodeManifestV3,
    EpisodeSampleV2,
    EpisodeStageV3,
    RobotDescriptorV2,
    TargetPoseV2,
)

_STOP = object()

class EpisodeWriter:
    """Encode frames off-thread and publish an episode only when complete."""

    def __init__(
        self,
        output_dir: Path,
        episode_id: str,
        seed: int,
        task: str,
        robot: RobotDescriptorV2,
        target: TargetPoseV2,
        sample_rate_hz: float,
        fixed_camera_name: str,
        wrist_camera_name: str,
        fixed_camera_2_name: str = "",
        queue_size: int = 8,
    ) -> None:
        self.episode_id = episode_id
        self.seed = seed
        self.task = task.strip()
        if not self.task:
            raise ValueError("episode task must not be empty")
        self.robot = robot
        self.target = target
        self.sample_rate_hz = float(sample_rate_hz)
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        self.camera_names = {
            "fixed": fixed_camera_name,
            "wrist": wrist_camera_name,
        }
        if fixed_camera_2_name:
            self.camera_names["fixed_2"] = fixed_camera_2_name
        self.started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.final_dir = output_dir / episode_id
        self.partial_dir = output_dir / f".{episode_id}.partial"
        if self.final_dir.exists() or self.partial_dir.exists():
            raise FileExistsError(f"episode path already exists for {episode_id!r}")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.partial_dir.mkdir()

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._sample_count = 0
        self._dimensions: dict[str, tuple[int, int]] | None = None
        self._completed_stages: list[EpisodeStageV3] = []
        self._active_stage: tuple[int, str] | None = None
        self._keyframe_indices = {0}
        self._thread = threading.Thread(
            target=self._run, name=f"episode-writer-{episode_id}", daemon=True
        )
        self._thread.start()

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def failed(self) -> bool:
        return self._error is not None

    def _set_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def fail(self, error: BaseException) -> None:
        self._set_error(error)

    def submit(
        self,
        sample: EpisodeSampleV2,
        fixed_rgb: Any,
        wrist_rgb: Any,
        fixed_2_rgb: Any = None,
    ) -> None:
        if self._error is not None:
            raise RuntimeError("episode writer has failed") from self._error
        if self.robot.gripper_mode == "binary" and (
            sample.gripper_command not in (0.0, 1.0)
            or sample.gripper_state not in (0.0, 1.0)
        ):
            raise ValueError("binary gripper command and state must be 0 or 1")
        frames = {"fixed": fixed_rgb, "wrist": wrist_rgb}
        if "fixed_2" in self.camera_names:
            if fixed_2_rgb is None:
                raise ValueError("fixed_2 RGB frame is required")
            frames["fixed_2"] = fixed_2_rgb
        elif fixed_2_rgb is not None:
            raise ValueError(
                "fixed_2 RGB frame was provided without a configured camera"
            )
        try:
            force_keyframe = sample.frame_index in self._keyframe_indices
            self._queue.put_nowait((sample, frames, force_keyframe))
        except queue.Full as exc:
            error = RuntimeError("episode writer queue filled; sample would be dropped")
            self._set_error(error)
            raise error from exc

    def active_stage_frame_count(self, next_frame_index: int) -> int | None:
        """Return captured frames in the active stage, or None before its first mark."""
        if self._error is not None:
            raise RuntimeError("episode writer has failed") from self._error
        if isinstance(next_frame_index, bool) or next_frame_index < 0:
            raise ValueError("next_frame_index must be a non-negative integer")
        if self._active_stage is None:
            return None
        return next_frame_index - self._active_stage[0]

    def stage_start_frames(self) -> tuple[int, ...]:
        """Return the frame indices that begin independently decodable stages."""
        starts = [stage.start_frame for stage in self._completed_stages]
        if self._active_stage is not None:
            starts.append(self._active_stage[0])
        return tuple(starts)

    def mark_stage(self, task: str, next_frame_index: int) -> dict[str, Any]:
        """Start a stage at the next frame, closing the previous stage."""
        if self._error is not None:
            raise RuntimeError("episode writer has failed") from self._error
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("stage task must not be empty")
        if isinstance(next_frame_index, bool) or next_frame_index < 0:
            raise ValueError("next_frame_index must be a non-negative integer")

        if self._active_stage is None:
            start_frame = 0
        else:
            start_frame, active_task = self._active_stage
            if next_frame_index <= start_frame:
                raise ValueError("cannot close a stage before it has captured a frame")
            self._completed_stages.append(
                EpisodeStageV3(start_frame, next_frame_index, active_task)
            )
            start_frame = next_frame_index
        self._keyframe_indices.add(start_frame)
        self._active_stage = (start_frame, normalized_task)
        return {"start_frame": start_frame, "task": normalized_task}

    def close(self, outcome: str) -> Path:
        while self._thread.is_alive():
            try:
                self._queue.put(_STOP, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            self._set_error(
                TimeoutError("episode writer did not close within 60 seconds")
            )
        if self._error is not None:
            raise RuntimeError(
                f"episode {self.episode_id} failed; partial data remains at {self.partial_dir}"
            ) from self._error
        if self._dimensions is None or self._sample_count == 0:
            raise RuntimeError("cannot publish an episode with no samples")

        assert self._dimensions is not None
        stages = list(self._completed_stages)
        if self._active_stage is not None:
            start_frame, task = self._active_stage
            if self._sample_count <= start_frame:
                raise RuntimeError(
                    "cannot publish an episode with an empty final stage"
                )
            stages.append(EpisodeStageV3(start_frame, self._sample_count, task))
        cameras = tuple(
            CameraStreamV2(
                role,
                component_name,
                f"{role}.mp4",
                self._dimensions[role][0],
                self._dimensions[role][1],
            )
            for role, component_name in self.camera_names.items()
        )
        manifest = EpisodeManifestV3(
            episode_id=self.episode_id,
            seed=self.seed,
            task=self.task,
            started_at=self.started_at,
            robot=self.robot,
            cameras=cameras,
            stages=tuple(stages),
            target=self.target,
            outcome=outcome,
            sample_count=self._sample_count,
            sample_rate_hz=self.sample_rate_hz,
        )
        manifest_path = self.partial_dir / "manifest.json"
        manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
        os.replace(self.partial_dir, self.final_dir)
        return self.final_dir

    def _run(self) -> None:
        outputs: dict[str, Any] = {}
        streams: dict[str, Any] = {}
        samples: list[EpisodeSampleV2] = []
        try:
            import av
            import pyarrow as pa
            import pyarrow.parquet as pq

            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                sample, frames, force_keyframe = item
                dimensions: dict[str, tuple[int, int]] = {}
                for role, rgb in frames.items():
                    shape = tuple(int(value) for value in rgb.shape)
                    if len(shape) != 3 or shape[2] != 3:
                        raise ValueError(
                            "episode RGB frames must have shape (height, width, 3)"
                        )
                    dimensions[role] = (shape[1], shape[0])
                if self._dimensions is None:
                    self._dimensions = dimensions
                    for role, (width, height) in dimensions.items():
                        output = av.open(
                            str(self.partial_dir / f"{role}.mp4"), mode="w"
                        )
                        stream = output.add_stream(
                            "libx264",
                            rate=Fraction(str(self.sample_rate_hz)),
                            options={"forced-idr": "1", "open-gop": "0"},
                        )
                        stream.width = width
                        stream.height = height
                        stream.pix_fmt = "yuv420p"
                        outputs[role] = output
                        streams[role] = stream
                elif dimensions != self._dimensions:
                    raise ValueError(
                        f"camera dimensions changed during episode: {self._dimensions} -> {dimensions}"
                    )

                for role, rgb in frames.items():
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    if force_keyframe:
                        frame.pict_type = av.video.frame.PictureType.I
                    for packet in streams[role].encode(frame):
                        outputs[role].mux(packet)
                samples.append(sample)
                self._sample_count += 1

            if not outputs:
                raise RuntimeError("episode writer received no frames")
            for role, output in outputs.items():
                for packet in streams[role].encode():
                    output.mux(packet)
                output.close()
            outputs.clear()

            vector6 = pa.list_(pa.float64(), 6)
            vector3 = pa.list_(pa.float64(), 3)
            vector4 = pa.list_(pa.float64(), 4)
            schema = pa.schema(
                [
                    ("frame_index", pa.int64()),
                    ("sim_time_s", pa.float64()),
                    ("joint_position_rad", vector6),
                    ("joint_velocity_rad_s", vector6),
                    ("joint_target_rad", vector6),
                    ("ee_position_m", vector3),
                    ("ee_rotation_wxyz", vector4),
                    ("gripper_command", pa.float64()),
                    ("gripper_state", pa.float64()),
                    ("joint_target_nudge_rad", vector6),
                ]
            )
            rows = [
                {
                    "frame_index": sample.frame_index,
                    "sim_time_s": sample.sim_time_s,
                    "joint_position_rad": sample.joint_position_rad,
                    "joint_velocity_rad_s": sample.joint_velocity_rad_s,
                    "joint_target_rad": sample.joint_target_rad,
                    "ee_position_m": sample.ee_position_m,
                    "ee_rotation_wxyz": sample.ee_rotation_wxyz,
                    "gripper_command": sample.gripper_command,
                    "gripper_state": sample.gripper_state,
                    "joint_target_nudge_rad": sample.joint_target_nudge_rad,
                }
                for sample in samples
            ]
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema),
                self.partial_dir / "samples.parquet",
                compression="zstd",
            )
        except BaseException as exc:
            self._set_error(exc)
        finally:
            for output in outputs.values():
                try:
                    output.close()
                except BaseException:
                    pass
