import asyncio
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import av
import numpy as np
import pytest

from examples import batch_picks


class FakeArm:
    def __init__(self):
        self.homes = 0

    async def move_to_joint_positions(self, positions):
        self.homes += 1


COSMOS_MODALITIES = ("rgb", "depth", "edges", "segmentation", "shaded_seg")


def _write_video(path, *, frame_count, width, height, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(
                np.full((height, width, 3), frame_index, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_cosmos_output(root, episode_id):
    dimensions = {"fixed": (16, 12), "wrist": (10, 8)}
    episode_path = root / episode_id
    for role, (width, height) in dimensions.items():
        for modality in COSMOS_MODALITIES:
            _write_video(
                episode_path / role / "clip_0000" / f"{modality}.mp4",
                frame_count=2,
                width=width,
                height=height,
                fps=15,
            )
    return episode_path, dimensions


def _snapshot_files(root):
    if not root.exists():
        return {}
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _args(
    tmp_path,
    count=3,
    *,
    cosmos_output=False,
    grip_only=False,
    grip_lift_m=0.05,
    noise_fraction=0.6,
    upload_dataset_id=None,
):
    return Namespace(
        address="localhost:8080",
        run_dir=str(tmp_path / "run"),
        count=count,
        start_seed=40,
        task="Pick up the block",
        world="sim-world",
        gripper="pick-gripper",
        motion="builtin",
        prop="pick_block",
        arm="pick-arm",
        fixed_camera="scene-cam",
        fixed_camera_2="",
        wrist_camera="wrist-cam",
        grasp_height_m=0.015,
        yaw_offset_deg=0.0,
        gripper_settle_sec=0.0,
        noise_fraction=noise_fraction,
        grip_only=grip_only,
        grip_lift_m=grip_lift_m,
        nudge_deg=0.0,
        nudge_interval_sec=3.0,
        nudge_duration_sec=0.6,
        colored_blocks=None,
        place_on=None,
        park_props=[],
        remote_episodes=False,
        episode_timeout_sec=0.0,
        cosmos_output=cosmos_output,
        upload_dataset_id=upload_dataset_id,
        upload_tag=["extra"],
        app_url=None,
    )


def test_batch_run_is_serial_and_resumes_finalized_episodes(tmp_path, monkeypatch):
    calls = []
    cosmos_output_dirs = []
    native_output_dirs = []
    episode_tasks = []
    episode_noise = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        calls.append(seed)
        cosmos_output_dirs.append(episode.cosmos_output_dir)
        native_output_dirs.append(episode.output_dir)
        episode_tasks.append(episode.task)
        episode_noise.append(config.noise)
        episode_dir = tmp_path / "run" / "episodes" / episode.episode_id
        episode_dir.mkdir()
        outcome = "failure" if seed == 41 else "success"
        (episode_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "seed": seed,
                    "outcome": outcome,
                    "sample_count": 12,
                    "sample_rate_hz": 15.0,
                }
            )
        )
        return {"outcome": outcome, "timing": {"wall_total_sec": 0.5}}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    arm = FakeArm()
    first = asyncio.run(
        batch_picks.generate_run(_args(tmp_path), object(), object(), object(), arm)
    )

    assert calls == [40, 41, 42]
    assert native_output_dirs == [str(tmp_path / "run" / "episodes")] * 3
    assert cosmos_output_dirs == [None, None, None]
    assert episode_tasks == ["Pick up the block"] * 3
    expected_noise = [
        batch_picks._episode_uses_noise(seed, 0.6) for seed in (40, 41, 42)
    ]
    assert episode_noise == expected_noise
    assert arm.homes == 3
    assert first["counts"] == {"success": 2, "failure": 1}
    assert [item["episode_id"] for item in first["attempts"]] == [
        "000000",
        "000001",
        "000002",
    ]
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["schema_version"] == 5
    assert run["config"]["cosmos_output"] is False
    assert run["config"]["noise_fraction"] == pytest.approx(0.6)
    assert "cosmos_output_dir" not in run["config"]
    assert [attempt["noise"] for attempt in first["attempts"]] == expected_noise
    assert all("cosmoswriter" not in attempt for attempt in first["attempts"])
    assert not (tmp_path / "run" / "cosmoswriter").exists()
    # additive per-attempt timing for measuring episode throughput
    for attempt in run["attempts"]:
        assert attempt["wall_time_sec"] >= attempt["home_move_sec"] >= 0.0
        assert attempt["started_at"].endswith("Z")
        assert attempt["result"]["timing"]["wall_total_sec"] >= 0.0

    second_arm = FakeArm()
    second = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path), object(), object(), object(), second_arm
        )
    )

    assert calls == [40, 41, 42]
    assert second_arm.homes == 0
    assert second["counts"] == first["counts"]

    expanded_arm = FakeArm()
    expanded = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=5), object(), object(), object(), expanded_arm
        )
    )

    assert calls == [40, 41, 42, 43, 44]
    assert expanded_arm.homes == 2
    assert len(expanded["attempts"]) == 5
    assert episode_noise == [
        batch_picks._episode_uses_noise(seed, 0.6)
        for seed in (40, 41, 42, 43, 44)
    ]
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["config"]["count"] == 5

    with pytest.raises(ValueError, match="resume configuration does not match"):
        asyncio.run(
            batch_picks.generate_run(
                _args(tmp_path, count=4), object(), object(), object(), FakeArm()
            )
        )



def test_resume_rejects_noise_fraction_change(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_picks, "_git_sha", lambda: "abc123")
    run_dir = tmp_path / "run"
    initial_config = {"count": 3, "noise_fraction": 0.6, "world": "sim-world"}
    batch_picks._load_or_create_run(run_dir, initial_config)

    with pytest.raises(ValueError, match="resume configuration does not match"):
        batch_picks._load_or_create_run(
            run_dir,
            {**initial_config, "count": 5, "noise_fraction": 0.4},
        )


@pytest.mark.parametrize(
    ("fraction", "expected"),
    ((0.0, False), (1.0, True)),
)
def test_episode_noise_fraction_boundaries(fraction, expected):
    assert batch_picks._episode_uses_noise(42, fraction) is expected


@pytest.mark.parametrize("value", (-0.1, 1.1, float("inf"), float("nan"), True))
def test_noise_fraction_must_be_finite_probability(value):
    with pytest.raises(ValueError, match="noise_fraction"):
        batch_picks._noise_fraction(value)

def test_batch_propagates_optional_second_fixed_camera(tmp_path, monkeypatch):
    args = _args(tmp_path, count=1)
    args.fixed_camera_2 = "third-cam"
    received = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        received.append(episode.fixed_camera_2_name)
        episode_dir = tmp_path / "run" / "episodes" / episode.episode_id
        episode_dir.mkdir()
        (episode_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "seed": seed,
                    "outcome": "success",
                    "sample_count": 1,
                    "sample_rate_hz": 15.0,
                }
            )
        )
        return {"outcome": "success"}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    asyncio.run(batch_picks.generate_run(args, object(), object(), object(), FakeArm()))

    assert received == ["third-cam"]
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["config"]["fixed_camera_2"] == "third-cam"


def test_grip_only_skips_home_and_persists_capture_config(tmp_path, monkeypatch):
    seen_configs = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        seen_configs.append(config)
        episode_dir = tmp_path / "run" / "episodes" / episode.episode_id
        episode_dir.mkdir()
        (episode_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "seed": seed,
                    "outcome": "success",
                    "sample_count": 2,
                    "sample_rate_hz": 15.0,
                }
            )
        )
        return {"outcome": "success"}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    arm = FakeArm()
    asyncio.run(
        batch_picks.generate_run(
            _args(
                tmp_path,
                count=1,
                grip_only=True,
                grip_lift_m=0.06,
                noise_fraction=0.0,
            ),
            object(),
            object(),
            object(),
            arm,
        )
    )

    assert arm.homes == 0
    assert seen_configs[0].grip_only is True
    assert seen_configs[0].grip_lift_m == pytest.approx(0.06)
    assert seen_configs[0].noise is False
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["config"]["grip_only"] is True
    assert run["config"]["grip_lift_m"] == pytest.approx(0.06)
    assert run["config"]["noise_fraction"] == pytest.approx(0.0)
    assert run["attempts"][0]["noise"] is False


def test_nudge_flags_reach_each_episode_and_persist_in_run_config(
    tmp_path, monkeypatch
):
    seen_episodes = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        seen_episodes.append(episode)
        episode_dir = tmp_path / "run" / "episodes" / episode.episode_id
        episode_dir.mkdir()
        (episode_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "seed": seed,
                    "outcome": "success",
                    "sample_count": 2,
                    "sample_rate_hz": 15.0,
                }
            )
        )
        return {"outcome": "success"}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    args = _args(tmp_path, count=1)
    args.nudge_deg = 5.0
    args.nudge_interval_sec = 1.2
    asyncio.run(batch_picks.generate_run(args, object(), object(), object(), FakeArm()))

    expected = {"amplitude_deg": 5.0, "interval_s": 1.2, "duration_s": 0.6}
    assert seen_episodes[0].nudge == expected
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["config"]["nudge"] == expected


def test_nudge_deg_takes_one_value_or_six_per_joint_values():
    assert batch_picks._nudge_deg("5") == 5.0
    assert batch_picks._nudge_deg("2.5,3.3,2.9,1.5,3.2,5") == [
        2.5, 3.3, 2.9, 1.5, 3.2, 5.0
    ]
    for bad in ("1,2", "-1", "a"):
        with pytest.raises(batch_picks.argparse.ArgumentTypeError):
            batch_picks._nudge_deg(bad)
    args = Namespace(nudge_deg=[0.0] * 6, nudge_interval_sec=1.2, nudge_duration_sec=0.6)
    assert batch_picks._nudge(args) is None


def test_colored_blocks_takes_two_label_prop_pairs():
    assert batch_picks._colored_blocks("red=pick_block,blue=blue_block") == {
        "red": "pick_block",
        "blue": "blue_block",
    }
    for bad in ("red=pick_block", "red=a,red=b", "red=a,blue=a", "red,blue=b"):
        with pytest.raises(batch_picks.argparse.ArgumentTypeError):
            batch_picks._colored_blocks(bad)


def test_colored_block_runs_pick_a_seeded_target_and_name_it_in_the_task(
    tmp_path, monkeypatch
):
    seen = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        seen.append((config, episode))
        return {"outcome": "failure"}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    args = _args(tmp_path, count=12)
    args.colored_blocks = {"red": "pick_block", "blue": "blue_block"}
    asyncio.run(batch_picks.generate_run(args, object(), object(), object(), FakeArm()))

    labels = set()
    for config, episode in seen:
        label = config.target_label
        labels.add(label)
        other = "blue" if label == "red" else "red"
        assert config.prop_name == args.colored_blocks[label]
        assert config.distractor_prop == args.colored_blocks[other]
        assert episode.task == f"pick up the {label} block"
    assert labels == {"red", "blue"}
    assert [config.target_label for config, _ in seen] == [
        batch_picks._target_block(seed, args.colored_blocks)[0]
        for seed in range(40, 52)
    ]
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["config"]["colored_blocks"] == args.colored_blocks
    assert [attempt["target_block"] for attempt in run["attempts"]] == [
        config.target_label for config, _ in seen
    ]


def test_place_on_takes_one_label_prop_pair():
    assert batch_picks._place_on("green=place_pad") == {"green": "place_pad"}
    for bad in ("green", "=place_pad", "green=", "green=a,blue=b"):
        with pytest.raises(batch_picks.argparse.ArgumentTypeError):
            batch_picks._place_on(bad)


def test_place_runs_set_the_pad_and_name_it_in_the_task(tmp_path, monkeypatch):
    seen = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        seen.append((config, episode))
        return {"outcome": "failure"}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    args = _args(tmp_path, count=2)
    args.place_on = {"green": "place_pad"}
    asyncio.run(batch_picks.generate_run(args, object(), object(), object(), FakeArm()))

    assert [(c.place_prop, c.place_label) for c, _ in seen] == [("place_pad", "green")] * 2
    assert {episode.task for _, episode in seen} == {"put the block on the green pad"}
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["config"]["place_on"] == {"green": "place_pad"}


class FakeWorld:
    def __init__(self):
        self.commands = []

    async def do_command(self, command):
        self.commands.append(command)
        return {}


def test_park_props_moves_unused_props_off_before_each_episode(tmp_path, monkeypatch):
    seen = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        seen.append(config)
        return {"outcome": "failure"}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    args = _args(tmp_path, count=6)
    args.colored_blocks = {"red": "pick_block", "blue": "blue_block"}
    args.park_props = ["blue_block", "place_pad"]
    world = FakeWorld()
    asyncio.run(batch_picks.generate_run(args, world, object(), object(), FakeArm()))

    # Both blocks are always in use for colored runs, so only the pad is parked.
    assert world.commands == [
        {"command": "park_props", "props": ["place_pad"]}
    ] * len(seen)
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["config"]["park_props"] == ["blue_block", "place_pad"]


def test_props_to_park_skips_the_props_an_episode_uses():
    config = batch_picks.PickConfig(place_prop="place_pad")
    assert batch_picks._props_to_park(["blue_block", "place_pad"], config) == [
        "blue_block"
    ]


def test_remote_episodes_record_the_sim_outcome_and_resume_without_rerecording(
    tmp_path, monkeypatch
):
    calls = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        calls.append(seed)
        return {
            "outcome": "success" if seed % 2 else "failure",
            "capture": {
                "episode_id": episode.episode_id,
                "outcome": "success" if seed % 2 else "failure",
                "path": f"/sim/{episode.episode_id}",
                "sample_count": 200,
            },
            "timing": {"wall_total_sec": 12.0},
        }

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    args = _args(tmp_path, count=3)
    args.remote_episodes = True
    asyncio.run(batch_picks.generate_run(args, object(), object(), object(), FakeArm()))

    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [a["status"] for a in run["attempts"]] == ["finalized"] * 3
    assert all(a["on_machine"] for a in run["attempts"])
    assert [a["outcome"] for a in run["attempts"]] == [
        "success" if seed % 2 else "failure" for seed in range(40, 43)
    ]
    assert run["attempts"][0]["path"] == "/sim/000000"
    assert run["attempts"][0]["result"]["timing"] == {"wall_total_sec": 12.0}

    # Extending the run records only the new episodes.
    calls.clear()
    args.count = 5
    asyncio.run(batch_picks.generate_run(args, object(), object(), object(), FakeArm()))
    assert calls == [43, 44]


def test_episode_timeout_closes_the_capture_and_moves_on(tmp_path, monkeypatch):
    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        if seed == 40:
            await asyncio.sleep(10)
        return {
            "outcome": "success",
            "capture": {
                "episode_id": episode.episode_id,
                "outcome": "success",
                "path": "/sim/x",
                "sample_count": 1,
            },
        }

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    args = _args(tmp_path, count=2)
    args.remote_episodes = True
    args.episode_timeout_sec = 0.05
    world = FakeWorld()
    asyncio.run(batch_picks.generate_run(args, world, object(), object(), FakeArm()))

    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [a["outcome"] for a in run["attempts"]] == ["incomplete", "success"]
    assert "TimeoutError" in run["attempts"][0]["error"]
    assert world.commands == [{"command": "stop_episode", "outcome": "incomplete"}]


class FakeUploadWorld:
    def __init__(self):
        self.commands = []

    async def do_command(self, command):
        self.commands.append(command)
        return {
            "dataset_id": command["upload"]["dataset_id"],
            "file_id": f"retry-{command['path'].rsplit('/', 1)[-1]}",
            "bytes": 10.0,
            "attempts": 1.0,
        }


def _capture(episode_id, seed, path, upload):
    return {
        "episode_id": episode_id,
        "seed": seed,
        "outcome": "success",
        "path": str(path),
        "sample_count": 12.0,
        "upload": upload,
    }


def test_batch_requests_module_upload_and_retries_failures_on_resume(
    tmp_path, monkeypatch
):
    stop_uploads = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        stop_uploads.append(episode.upload)
        episode_dir = tmp_path / "run" / "episodes" / episode.episode_id
        episode_dir.mkdir()
        (episode_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "seed": seed,
                    "outcome": "success",
                    "sample_count": 12,
                    "sample_rate_hz": 15.0,
                }
            )
        )
        upload = (
            {"dataset_id": "dataset-1", "error": "RuntimeError: down", "attempts": 3.0}
            if episode.episode_id == "000000"
            else {"dataset_id": "dataset-1", "file_id": "file-1", "bytes": 10.0}
        )
        return {
            "outcome": "success",
            "capture": _capture(episode.episode_id, seed, episode_dir, upload),
        }

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    args = _args(tmp_path, count=2, upload_dataset_id="dataset-1")
    world = FakeUploadWorld()

    first = asyncio.run(
        batch_picks.generate_run(args, world, object(), object(), FakeArm())
    )

    assert (
        stop_uploads
        == [{"dataset_id": "dataset-1", "tags": ["run", "isaac-episode", "extra"]}] * 2
    )
    assert world.commands == []
    assert first["upload_errors"] == ["000000"]
    assert first["attempts"][0]["upload"]["attempts"] == 3
    assert first["attempts"][1]["upload"] == {
        "dataset_id": "dataset-1",
        "file_id": "file-1",
        "bytes": 10,
    }

    second = asyncio.run(
        batch_picks.generate_run(args, world, object(), object(), FakeArm())
    )

    assert len(stop_uploads) == 2
    assert world.commands == [
        {
            "command": "upload_episode",
            "path": str(tmp_path / "run" / "episodes" / "000000"),
            "upload": {
                "dataset_id": "dataset-1",
                "tags": ["run", "isaac-episode", "extra"],
            },
        }
    ]
    assert "upload_errors" not in second
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [a["upload"]["file_id"] for a in run["attempts"]] == [
        "retry-000000",
        "file-1",
    ]


def test_batch_records_episodes_stored_only_on_machine(tmp_path, monkeypatch):
    calls = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        calls.append(episode.episode_id)
        machine_path = f"/machine/runs/run/episodes/{episode.episode_id}"
        upload = {"dataset_id": "dataset-1", "file_id": "file-1", "bytes": 10.0}
        return {
            "outcome": "success",
            "capture": _capture(episode.episode_id, seed, machine_path, upload),
        }

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    world = FakeUploadWorld()

    first = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=1, upload_dataset_id="dataset-1"),
            world,
            object(),
            object(),
            FakeArm(),
        )
    )
    second = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=2, upload_dataset_id="dataset-1"),
            world,
            object(),
            object(),
            FakeArm(),
        )
    )

    attempt = first["attempts"][0]
    assert attempt["status"] == "finalized"
    assert attempt["on_machine"] is True
    assert attempt["noise"] is False
    assert attempt["sample_count"] == 12
    assert attempt["path"] == "/machine/runs/run/episodes/000000"
    assert calls == ["000000", "000001"]
    assert second["counts"] == {"success": 2}
    assert world.commands == []


def test_batch_rejects_machine_result_for_wrong_episode(tmp_path, monkeypatch):
    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        return {"outcome": "success", "capture": {"episode_id": "other"}}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)

    result = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=1), object(), object(), object(), FakeArm()
        )
    )

    attempt = result["attempts"][0]
    assert attempt["status"] == "error"
    assert "did not report finalized episode 000000" in attempt["error"]


def test_batch_requires_schema_4_and_matching_resume_config(tmp_path, monkeypatch):
    args = _args(tmp_path, count=1)

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        episode_dir = tmp_path / "run" / "episodes" / episode.episode_id
        episode_dir.mkdir(parents=True)
        (episode_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "seed": seed,
                    "outcome": "success",
                    "sample_count": 12,
                    "sample_rate_hz": 15.0,
                }
            )
        )
        return {"outcome": "success"}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    asyncio.run(batch_picks.generate_run(args, object(), object(), object(), FakeArm()))

    run_path = tmp_path / "run" / "run.json"
    run = json.loads(run_path.read_text())
    run["schema_version"] = 2
    run_path.write_text(json.dumps(run))

    with pytest.raises(ValueError, match="unsupported run schema_version"):
        asyncio.run(
            batch_picks.generate_run(args, object(), object(), object(), FakeArm())
        )

    run["schema_version"] = 5
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="resume configuration does not match"):
        asyncio.run(
            batch_picks.generate_run(
                _args(tmp_path, count=1, cosmos_output=True),
                object(),
                object(),
                object(),
                FakeArm(),
            )
        )


def test_failed_attempt_records_wall_time(tmp_path, monkeypatch):
    async def failing_execute(*unused_args, **unused_kwargs):
        raise RuntimeError("motion service could not move pick-gripper")

    monkeypatch.setattr(batch_picks, "execute_pick", failing_execute)
    result = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=1), object(), object(), object(), FakeArm()
        )
    )

    attempt = result["attempts"][0]
    assert attempt["status"] == "error"
    assert attempt["wall_time_sec"] >= attempt["home_move_sec"] >= 0.0
    assert "started_at" in attempt


def test_batch_run_preserves_partial_episode_without_retry(tmp_path, monkeypatch):
    args = _args(tmp_path, count=1)
    partial = tmp_path / "run" / "episodes" / ".000000.partial"
    partial.mkdir(parents=True)

    async def should_not_execute(*unused_args, **unused_kwargs):
        raise AssertionError("partial episode must not be overwritten")

    monkeypatch.setattr(batch_picks, "execute_pick", should_not_execute)
    arm = FakeArm()
    result = asyncio.run(
        batch_picks.generate_run(args, object(), object(), object(), arm)
    )

    assert arm.homes == 0
    assert result["counts"] == {"incomplete": 1}
    assert result["attempts"][0]["status"] == "partial"


def test_batch_persists_sibling_cosmoswriter_streams_and_resumes_valid_episode(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    cosmos_root = run_dir / "cosmoswriter"
    received_configs = []

    async def fake_execute(world, gripper, motion, arm, seed, config, episode):
        received_configs.append(episode)
        episode_dir = run_dir / "episodes" / episode.episode_id
        episode_dir.mkdir(parents=True)
        (episode_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "seed": seed,
                    "outcome": "success",
                    "sample_count": 2,
                    "sample_rate_hz": 15.0,
                }
            )
        )
        _write_cosmos_output(cosmos_root, episode.episode_id)
        return {"outcome": "success"}

    monkeypatch.setattr(batch_picks, "execute_pick", fake_execute)
    asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=1, cosmos_output=True),
            object(),
            object(),
            object(),
            FakeArm(),
        )
    )

    assert received_configs[0].output_dir == str(run_dir / "episodes")
    assert received_configs[0].cosmos_output_dir == str(cosmos_root / "000000")
    run = json.loads((run_dir / "run.json").read_text())
    assert run["config"]["cosmos_output"] is True
    assert "cosmos_output_dir" not in run["config"]
    attempt = run["attempts"][0]
    assert attempt["status"] == "finalized"
    cosmoswriter = attempt["cosmoswriter"]
    assert cosmoswriter["path"] == str(cosmos_root / "000000")
    assert set(cosmoswriter["roles"]) == {"fixed", "wrist"}
    expected_dimensions = {"fixed": (16, 12), "wrist": (10, 8)}
    for role, (width, height) in expected_dimensions.items():
        streams = cosmoswriter["roles"][role]
        assert set(streams) == set(COSMOS_MODALITIES)
        for modality in COSMOS_MODALITIES:
            stream = streams[modality]
            assert stream["relative_path"] == f"{role}/clip_0000/{modality}.mp4"
            assert stream["frame_count"] == 2
            assert (stream["width"], stream["height"]) == (width, height)
            assert stream["codec"] == "h264"
            assert stream["pixel_format"] == "yuv420p"
            assert stream["average_fps"] == 15.0
            assert stream["raw_average_fps"] == 15.0
            assert stream["timestamp_spacing_s"] == pytest.approx(1.0 / 15.0)
            assert stream["duration_s"] == pytest.approx(2.0 / 15.0)

    async def should_not_execute(*unused_args, **unused_kwargs):
        raise AssertionError("valid finalized episode must not be recaptured")

    monkeypatch.setattr(batch_picks, "execute_pick", should_not_execute)
    resumed_arm = FakeArm()
    resumed = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=1, cosmos_output=True),
            object(),
            object(),
            object(),
            resumed_arm,
        )
    )

    assert resumed_arm.homes == 0
    assert resumed["attempts"][0]["cosmoswriter"] == cosmoswriter

    with pytest.raises(ValueError, match="resume configuration does not match"):
        asyncio.run(
            batch_picks.generate_run(
                _args(tmp_path, count=1, cosmos_output=False),
                object(),
                object(),
                object(),
                FakeArm(),
            )
        )


@pytest.mark.parametrize("kind", ("missing", "undecodable"))
def test_batch_records_cosmos_validation_error_without_recapturing_native_episode(
    tmp_path, monkeypatch, kind
):
    cosmos_root = tmp_path / "run" / "cosmoswriter"
    episode_dir = tmp_path / "run" / "episodes" / "000000"
    episode_dir.mkdir(parents=True)
    manifest_path = episode_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "000000",
                "seed": 40,
                "outcome": "success",
                "sample_count": 2,
                "sample_rate_hz": 15.0,
            }
        )
    )
    if kind == "undecodable":
        cosmos_path, _ = _write_cosmos_output(cosmos_root, "000000")
        (cosmos_path / "fixed" / "clip_0000" / "depth.mp4").write_bytes(b"not an MP4")

    before_manifest = manifest_path.read_bytes()
    before_cosmos = _snapshot_files(cosmos_root)

    async def should_not_execute(*unused_args, **unused_kwargs):
        raise AssertionError("existing native episode must not be recaptured")

    monkeypatch.setattr(batch_picks, "execute_pick", should_not_execute)
    arm = FakeArm()
    result = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=1, cosmos_output=True),
            object(),
            object(),
            object(),
            arm,
        )
    )

    attempt = result["attempts"][0]
    assert arm.homes == 0
    assert result["counts"] == {"incomplete": 1}
    assert attempt["status"] == "error"
    assert attempt["outcome"] == "incomplete"
    assert attempt["native_outcome"] == "success"
    assert str(cosmos_root / "000000") in attempt["error"]
    assert manifest_path.read_bytes() == before_manifest
    assert _snapshot_files(cosmos_root) == before_cosmos


def test_batch_preserves_sibling_cosmos_collision_without_native_episode(
    tmp_path, monkeypatch
):
    cosmos_episode = tmp_path / "run" / "cosmoswriter" / "000000"
    cosmos_episode.mkdir(parents=True)
    marker = cosmos_episode / "operator-owned.txt"
    marker.write_text("preserve me")

    async def should_not_execute(*unused_args, **unused_kwargs):
        raise AssertionError(
            "native capture must not overwrite a CosmosWriter collision"
        )

    monkeypatch.setattr(batch_picks, "execute_pick", should_not_execute)
    arm = FakeArm()
    result = asyncio.run(
        batch_picks.generate_run(
            _args(tmp_path, count=1, cosmos_output=True),
            object(),
            object(),
            object(),
            arm,
        )
    )

    attempt = result["attempts"][0]
    assert arm.homes == 0
    assert attempt["status"] == "error"
    assert attempt["path"] is None
    assert str(cosmos_episode) in attempt["error"]
    assert marker.read_text() == "preserve me"


def test_batch_cli_uses_boolean_cosmos_output_and_rejects_selected_root(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_picks.py",
            "localhost:8080",
            str(tmp_path / "run"),
            "--count",
            "1",
            "--cosmos-output",
            "--upload-dataset-id",
            "dataset-1",
            "--upload-tag",
            "extra",
            "--grip-only",
            "--grip-lift-m",
            "0.06",
        ],
    )

    args = batch_picks._parse_args()

    assert args.cosmos_output is True
    assert args.upload_dataset_id == "dataset-1"
    assert args.upload_tag == ["extra"]
    assert not hasattr(args, "cosmos_output_dir")
    assert args.grip_only is True
    assert args.grip_lift_m == pytest.approx(0.06)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_picks.py",
            "localhost:8080",
            str(tmp_path / "run"),
            "--count",
            "1",
            "--cosmos-output-dir",
            str(tmp_path / "cosmos"),
        ],
    )
    with pytest.raises(SystemExit):
        batch_picks._parse_args()


def test_batch_picks_help_runs_as_direct_script():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/batch_picks.py", "--help"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--cosmos-output" in result.stdout
    assert "--noise-fraction" in result.stdout
    assert "--no-noise" not in result.stdout
    assert "--grip-only" in result.stdout
    assert "--grip-lift-m" in result.stdout
    assert "--upload-dataset-id" in result.stdout
