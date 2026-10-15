import asyncio
import itertools
import json
import threading
import time
from pathlib import Path

import av
import pyarrow.parquet as pq
import pytest
from PIL import Image
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import (
    ArmHandle,
    CameraHandle,
    GripperHandle,
    SimConfig,
    SimManager,
)


def _idr_frame_indices(path: Path) -> set[int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame_by_pts = {
            frame.pts: index for index, frame in enumerate(container.decode(stream))
        }
        extradata = stream.codec_context.extradata
        nal_length_size = (extradata[4] & 0x03) + 1

    result: set[int] = set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.pts not in frame_by_pts:
                continue
            payload = bytes(packet)
            offset = 0
            while offset < len(payload):
                nal_size = int.from_bytes(
                    payload[offset : offset + nal_length_size], "big"
                )
                offset += nal_length_size
                if payload[offset] & 0x1F == 5:
                    result.add(frame_by_pts[packet.pts])
                offset += nal_size
    return result


def _config(name, attrs):
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


class FakePhysicsWorld:
    def __init__(self, *, render_delivery=0, render_png_frames=None):
        self.callbacks = {}
        self.current_time = 0.0
        self.events = []
        self._render_delivery = render_delivery
        self._render_png_frames = render_png_frames
        self._render_writer = None

    def add_physics_callback(self, name, callback):
        self.callbacks[name] = callback
        self.events.append(("add_callback", name, threading.get_ident()))

    def remove_physics_callback(self, name):
        self.callbacks.pop(name)
        self.events.append(("remove_callback", name, threading.get_ident()))

    def set_render_writer(self, writer):
        self._render_writer = writer

    def render(self):
        self.events.append(("render", threading.get_ident()))
        if self._render_writer is not None:
            self._render_writer.deliver(self._render_delivery, self._render_png_frames)


COSMOS_MODALITIES = ("rgb", "depth", "edges", "segmentation", "shaded_seg")


def _write_cosmos_png_authority(root, frame_count):
    dimensions = {"fixed": (16, 12), "wrist": (12, 8)}
    for role, (width, height) in dimensions.items():
        for modality in COSMOS_MODALITIES:
            image_dir = root / role / "clip_0000" / modality
            image_dir.mkdir(parents=True, exist_ok=True)
            for ordinal in range(frame_count):
                Image.new("RGB", (width, height), color=(ordinal * 70, 0, 0)).save(
                    image_dir / f"{modality}_{ordinal:04d}.png"
                )


class FakeCosmosWriter:
    def __init__(self, output_dir, events, finalization_error=None):
        self.output_dir = output_dir
        self.events = events
        self.finalization_error = finalization_error
        self.num_written = 0
        self.schedule_calls = 0

    def schedule_write(self):
        self.schedule_calls += 1
        self.events.append(("schedule_write", threading.get_ident()))

    def deliver(self, count, png_frame_count=None):
        self.num_written += count
        self.events.append(("deliver", count, threading.get_ident()))
        _write_cosmos_png_authority(
            self.output_dir,
            self.num_written if png_frame_count is None else png_frame_count,
        )

    def on_final_frame(self):
        self.events.append(("on_final_frame", threading.get_ident()))
        if self.finalization_error is not None:
            raise self.finalization_error

    def detach(self):
        self.events.append(("detach", threading.get_ident()))


class FakeRenderProduct:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def destroy(self):
        self.events.append(("destroy", self.name, threading.get_ident()))


def _start_fake_cosmos_capture(manager, world, writer, products, png_frame_count=None):
    def start(*unused_args):
        writer.schedule_write()
        manager._cosmos_scheduled_count = 1
        writer.deliver(1, png_frame_count)
        world.set_render_writer(writer)
        return writer, products

    return start


def test_mock_episode_has_aligned_samples_and_video(tmp_path):
    previous = SimManager._instance
    manager = SimManager()
    SimManager._instance = manager
    thread = threading.Thread(target=manager.main_loop, daemon=True)
    thread.start()
    try:
        world = IsaacWorld.new(
            _config("sim-world", {"mock": True, "physics_dt": 1.0 / 60.0}), {}
        )
        IsaacArm.new(
            _config(
                "pick-arm",
                {
                    "world": "sim-world",
                    "asset": "lite6",
                    "mock_dof": 6,
                },
            ),
            {},
        )
        IsaacGripper.new(
            _config(
                "pick-gripper",
                {"world": "sim-world", "arm": "pick-arm", "asset": "lite6_gripper"},
            ),
            {},
        )
        IsaacCamera.new(
            _config(
                "scene-cam",
                {"world": "sim-world", "width": 64, "height": 48},
            ),
            {},
        )
        IsaacCamera.new(
            _config(
                "wrist-cam",
                {"world": "sim-world", "width": 32, "height": 24},
            ),
            {},
        )
        IsaacCamera.new(
            _config(
                "third-cam",
                {"world": "sim-world", "width": 40, "height": 30},
            ),
            {},
        )

        start = asyncio.run(
            world.do_command(
                {
                    "command": "start_episode",
                    "output_dir": str(tmp_path),
                    "episode_id": "000042",
                    "seed": 42,
                    "task": "Pick up the block",
                    "arm": "pick-arm",
                    "gripper": "pick-gripper",
                    "fixed_camera": "scene-cam",
                    "fixed_camera_2": "third-cam",
                    "wrist_camera": "wrist-cam",
                    "target": {
                        "prop_name": "pick_block",
                        "position_m": [0.6, 0.0, 0.0075],
                        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                }
            )
        )
        assert start["sample_rate_hz"] == 15.0
        first_stage = asyncio.run(
            world.do_command(
                {
                    "command": "mark_episode_stage",
                    "task": "Center the gripper above the block",
                }
            )
        )
        time.sleep(0.12)
        second_stage = asyncio.run(
            world.do_command(
                {
                    "command": "mark_episode_stage",
                    "task": "Lower the gripper to grasp the block",
                }
            )
        )
        time.sleep(0.12)
        stopped = asyncio.run(
            world.do_command(
                {
                    "command": "stop_episode",
                    "outcome": "success",
                }
            )
        )
    finally:
        manager.request_stop()
        thread.join(timeout=5)
        SimManager._instance = previous

    episode = tmp_path / "000042"
    assert stopped["path"] == str(episode)
    assert not (tmp_path / ".000042.partial").exists()
    manifest = json.loads((episode / "manifest.json").read_text())
    samples = pq.read_table(episode / "samples.parquet").to_pylist()
    assert manifest["sample_count"] == stopped["sample_count"] == len(samples)
    assert manifest["schema_version"] == 3
    assert manifest["sample_rate_hz"] == 15.0
    assert manifest["task"] == "Pick up the block"
    assert first_stage["start_frame"] == 0
    assert second_stage["start_frame"] >= 33
    assert manifest["stages"] == [
        {
            "start_frame": 0,
            "end_frame_exclusive": second_stage["start_frame"],
            "task": "Center the gripper above the block",
        },
        {
            "start_frame": second_stage["start_frame"],
            "end_frame_exclusive": manifest["sample_count"],
            "task": "Lower the gripper to grasp the block",
        },
    ]
    assert all(
        stage["end_frame_exclusive"] - stage["start_frame"] >= 33
        for stage in manifest["stages"]
    )
    assert manifest["robot"]["arm_model"] == "lite6"
    assert manifest["robot"]["joint_names"] == [
        f"joint{index}" for index in range(1, 7)
    ]
    assert manifest["robot"]["gripper_model"] == "lite6_gripper"
    assert manifest["robot"]["gripper_mode"] == "binary"
    assert all(sample["gripper_command"] in (0.0, 1.0) for sample in samples)
    assert all(sample["gripper_state"] in (0.0, 1.0) for sample in samples)
    assert len(samples) >= 3
    assert [sample["frame_index"] for sample in samples] == list(range(len(samples)))
    assert [
        later["sim_time_s"] - earlier["sim_time_s"]
        for earlier, later in itertools.pairwise(samples)
    ] == pytest.approx([1.0 / 15.0] * (len(samples) - 1))
    assert all(len(sample["joint_position_rad"]) == 6 for sample in samples)

    assert {camera["role"] for camera in manifest["cameras"]} == {
        "fixed",
        "fixed_2",
        "wrist",
    }
    stage_starts = {stage["start_frame"] for stage in manifest["stages"]}
    for name in ("fixed.mp4", "fixed_2.mp4", "wrist.mp4"):
        with av.open(str(episode / name), mode="r") as container:
            assert sum(1 for _ in container.decode(video=0)) == len(samples)
        assert stage_starts <= _idr_frame_indices(episode / name)



def test_cosmos_capture_rejects_mock_mode(tmp_path):
    manager = SimManager()
    manager.mock = True
    manager.cfg = SimConfig(mock=True)

    with pytest.raises(RuntimeError, match="unavailable in mock mode"):
        manager.start_episode(
            None,
            "000042",
            42,
            "Pick up the block",
            "pick-arm",
            "pick-gripper",
            "scene-cam",
            "wrist-cam",
            {},
            str(tmp_path),
        )


def _start_cosmos_only_capture(
    tmp_path,
    *,
    render_delivery=0,
    render_png_frames=None,
    initial_png_frames=None,
    finalization_error=None,
):
    class FakeCamera(CameraHandle):
        pass

    manager = SimManager()
    manager.cfg = SimConfig()
    manager._sim_thread_id = threading.get_ident()
    world = FakePhysicsWorld(
        render_delivery=render_delivery, render_png_frames=render_png_frames
    )
    manager.world = world
    manager._handles["scene-cam"] = ({}, FakeCamera())
    manager._handles["wrist-cam"] = ({}, FakeCamera())
    writer = FakeCosmosWriter(tmp_path, world.events, finalization_error)
    products = (
        FakeRenderProduct("fixed", world.events),
        FakeRenderProduct("wrist", world.events),
    )
    manager._start_cosmos_captures = _start_fake_cosmos_capture(
        manager, world, writer, products, initial_png_frames
    )
    started = manager.start_episode(
        None,
        "000042",
        42,
        "Pick up the block",
        "pick-arm",
        "pick-gripper",
        "scene-cam",
        "wrist-cam",
        {},
        str(tmp_path),
    )
    return manager, world, writer, started


def test_cosmos_flushes_one_pending_trigger_before_finalization(tmp_path, monkeypatch):
    class FakeCamera(CameraHandle):
        def capture_rgb(self):
            return None

    class FakeArm(ArmHandle):
        def joint_names(self):
            return [f"joint{index}" for index in range(1, 7)]

        def capture_state(self):
            return (
                [0.0] * 6,
                [0.0] * 6,
                [0.0] * 6,
                [0.0] * 3,
                [1.0, 0.0, 0.0, 0.0],
            )

    class FakeGripper(GripperHandle):
        def capture_state(self):
            return 0.0, 0.0

    class FakeEpisodeWriter:
        failed = False

        def __init__(self, output_dir, *unused_args):
            self.partial_dir = Path(output_dir) / ".000042.partial"
            self.sample_count = 0

        def submit(self, *unused_args):
            self.sample_count += 1

        def fail(self, unused_error):
            self.failed = True

        def stage_start_frames(self):
            return (0,)

        def close(self, outcome):
            world.events.append(("native_close", outcome, threading.get_ident()))
            return tmp_path / "000042"

    manager = SimManager()
    manager.cfg = SimConfig()
    manager._sim_thread_id = threading.get_ident()
    world = FakePhysicsWorld(render_delivery=1)
    manager.world = world
    manager._handles.update(
        {
            "pick-arm": ({"asset": "arm"}, FakeArm()),
            "pick-gripper": ({"asset": "gripper"}, FakeGripper()),
            "scene-cam": ({}, FakeCamera()),
            "wrist-cam": ({}, FakeCamera()),
        }
    )
    monkeypatch.setattr("isaac_module.episode_capture.EpisodeWriter", FakeEpisodeWriter)
    cosmos_writer = FakeCosmosWriter(tmp_path / "cosmos", world.events)
    products = (
        FakeRenderProduct("fixed", world.events),
        FakeRenderProduct("wrist", world.events),
    )
    manager._start_cosmos_captures = _start_fake_cosmos_capture(
        manager, world, cosmos_writer, products
    )

    started = manager.start_episode(
        str(tmp_path / "native"),
        "000042",
        42,
        "Pick up the block",
        "pick-arm",
        "pick-gripper",
        "scene-cam",
        "wrist-cam",
        {
            "prop_name": "pick_block",
            "position_m": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        str(tmp_path / "cosmos"),
    )
    world.callbacks["episode_capture"](1.0 / 15.0)
    world.callbacks["cosmos_capture"](1.0 / 15.0)
    assert cosmos_writer.schedule_calls == 2
    assert cosmos_writer.num_written == 1

    world.events.clear()
    stopped = manager.stop_episode("success")

    assert started["cosmos_partial_paths"] == {
        "fixed": str(tmp_path / "cosmos" / "fixed" / "clip_0000"),
        "wrist": str(tmp_path / "cosmos" / "wrist" / "clip_0000"),
    }
    assert stopped["cosmos_paths"] == {
        "fixed": str(tmp_path / "cosmos" / "fixed" / "clip_0000"),
        "wrist": str(tmp_path / "cosmos" / "wrist" / "clip_0000"),
    }
    assert stopped["sample_count"] == cosmos_writer.num_written == 2
    assert [event[0] for event in world.events] == [
        "remove_callback",
        "remove_callback",
        "render",
        "deliver",
        "on_final_frame",
        "detach",
        "destroy",
        "destroy",
        "native_close",
    ]
    assert [event[1] for event in world.events[:2]] == [
        "episode_capture",
        "cosmos_capture",
    ]
    assert {event[-1] for event in world.events} == {threading.get_ident()}
    for role in ("fixed", "wrist"):
        for modality in COSMOS_MODALITIES:
            image_dir = tmp_path / "cosmos" / role / "clip_0000" / modality
            assert (image_dir / f"{modality}_0001.png").is_file()
            with av.open(
                str(tmp_path / "cosmos" / role / "clip_0000" / f"{modality}.mp4")
            ) as video:
                assert sum(1 for _ in video.decode(video=0)) == 2


def test_cosmos_skips_render_when_schedules_are_drained(tmp_path):
    manager, world, writer, started = _start_cosmos_only_capture(tmp_path)

    world.events.clear()
    stopped = manager.stop_episode("success")

    assert started["cosmos_partial_paths"] == {
        "fixed": str(tmp_path / "fixed" / "clip_0000"),
        "wrist": str(tmp_path / "wrist" / "clip_0000"),
    }
    assert stopped["cosmos_paths"] == {
        "fixed": str(tmp_path / "fixed" / "clip_0000"),
        "wrist": str(tmp_path / "wrist" / "clip_0000"),
    }
    assert writer.num_written == 1
    assert [event[0] for event in world.events] == [
        "remove_callback",
        "on_final_frame",
        "detach",
        "destroy",
        "destroy",
    ]
    assert all(event[-1] == threading.get_ident() for event in world.events)
    for role in ("fixed", "wrist"):
        for modality in COSMOS_MODALITIES:
            with av.open(
                str(tmp_path / role / "clip_0000" / f"{modality}.mp4")
            ) as video:
                assert sum(1 for _ in video.decode(video=0)) == 1


def test_cosmos_rejects_more_than_one_pending_trigger_after_cleanup(tmp_path):
    manager, world, writer, _ = _start_cosmos_only_capture(
        tmp_path, initial_png_frames=3
    )
    world.callbacks["cosmos_capture"](1.0 / 15.0)
    world.callbacks["cosmos_capture"](1.0 / 15.0)
    assert writer.schedule_calls == 3

    world.events.clear()
    with pytest.raises(RuntimeError, match="CosmosWriter synchronization failed"):
        manager.stop_episode("failure")

    assert writer.num_written == 1
    assert [event[0] for event in world.events] == [
        "remove_callback",
        "on_final_frame",
        "detach",
        "destroy",
        "destroy",
    ]
    assert not list(tmp_path.rglob("*.mp4"))


@pytest.mark.parametrize(
    ("render_delivery", "expected_num_written"),
    [(0, 1), (2, 3)],
    ids=("render-no-delivery", "render-over-delivery"),
)
def test_cosmos_rejects_underdelivery_or_overdelivery_after_render(
    tmp_path, render_delivery, expected_num_written
):
    manager, world, writer, _ = _start_cosmos_only_capture(
        tmp_path,
        render_delivery=render_delivery,
        render_png_frames=2,
        initial_png_frames=2,
    )
    world.callbacks["cosmos_capture"](1.0 / 15.0)
    assert writer.schedule_calls == 2

    world.events.clear()
    with pytest.raises(RuntimeError, match="CosmosWriter synchronization failed"):
        manager.stop_episode("failure")

    assert writer.num_written == expected_num_written
    assert [event[0] for event in world.events] == [
        "remove_callback",
        "render",
        "deliver",
        "on_final_frame",
        "detach",
        "destroy",
        "destroy",
    ]
    assert all(event[-1] == threading.get_ident() for event in world.events)
    assert not list(tmp_path.rglob("*.mp4"))


def test_cosmos_capture_rejects_native_count_mismatch_after_cleanup(
    tmp_path, monkeypatch
):
    class FakeCamera(CameraHandle):
        def capture_rgb(self):
            return None

    class FakeArm(ArmHandle):
        def joint_names(self):
            return [f"joint{index}" for index in range(1, 7)]

        def capture_state(self):
            return (
                [0.0] * 6,
                [0.0] * 6,
                [0.0] * 6,
                [0.0] * 3,
                [1.0, 0.0, 0.0, 0.0],
            )

    class FakeGripper(GripperHandle):
        def capture_state(self):
            return 0.0, 0.0

    class FakeEpisodeWriter:
        failed = False

        def __init__(self, output_dir, *unused_args):
            self.partial_dir = Path(output_dir) / ".000042.partial"
            self.sample_count = 0

        def submit(self, *unused_args):
            self.sample_count += 1

        def fail(self, unused_error):
            self.failed = True

        def stage_start_frames(self):
            return (0,)

        def close(self, outcome):
            world.events.append(("native_close", outcome, threading.get_ident()))
            return tmp_path / "000042"

    manager = SimManager()
    manager.cfg = SimConfig()
    manager._sim_thread_id = threading.get_ident()
    world = FakePhysicsWorld()
    manager.world = world
    manager._handles.update(
        {
            "pick-arm": ({"asset": "arm"}, FakeArm()),
            "pick-gripper": ({"asset": "gripper"}, FakeGripper()),
            "scene-cam": ({}, FakeCamera()),
            "wrist-cam": ({}, FakeCamera()),
        }
    )
    monkeypatch.setattr("isaac_module.episode_capture.EpisodeWriter", FakeEpisodeWriter)
    cosmos_writer = FakeCosmosWriter(tmp_path / "cosmos", world.events)
    products = (
        FakeRenderProduct("fixed", world.events),
        FakeRenderProduct("wrist", world.events),
    )
    manager._start_cosmos_captures = _start_fake_cosmos_capture(
        manager, world, cosmos_writer, products
    )
    manager.start_episode(
        str(tmp_path / "native"),
        "000042",
        42,
        "Pick up the block",
        "pick-arm",
        "pick-gripper",
        "scene-cam",
        "wrist-cam",
        {
            "prop_name": "pick_block",
            "position_m": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        str(tmp_path / "cosmos"),
    )
    world.callbacks["cosmos_capture"](1.0 / 15.0)
    cosmos_writer.deliver(1)

    world.events.clear()
    with pytest.raises(RuntimeError, match="native sample_count 1"):
        manager.stop_episode("success")

    assert cosmos_writer.num_written == 2
    assert [event[0] for event in world.events] == [
        "remove_callback",
        "remove_callback",
        "on_final_frame",
        "detach",
        "destroy",
        "destroy",
        "native_close",
    ]
    assert not list((tmp_path / "cosmos").rglob("*.mp4"))


def test_cosmos_finalization_failure_is_explicit_and_cleans_up(tmp_path):
    manager, world, writer, _ = _start_cosmos_only_capture(
        tmp_path, finalization_error=OSError("encoder failed")
    )

    world.events.clear()
    with pytest.raises(
        RuntimeError, match="CosmosWriter finalization failed"
    ) as exc_info:
        manager.stop_episode("failure")

    assert isinstance(exc_info.value.__cause__, OSError)
    assert writer.num_written == 1
    assert [event[0] for event in world.events] == [
        "remove_callback",
        "on_final_frame",
        "detach",
        "destroy",
        "destroy",
    ]
    assert not list(tmp_path.rglob("*.mp4"))
