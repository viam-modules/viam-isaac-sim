"""Replaying recorded states: world DoCommands off the event loop, and a
lockstep `step` while paused."""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimManager


def test_world_prop_commands_do_not_block_the_event_loop():
    # Each call waits for the other at a barrier: run on the event loop,
    # the first would block it and the second could never start.
    barrier = threading.Barrier(3, timeout=5.0)

    class BlockingSim:
        def get_prop_pose(self, prop):
            barrier.wait()
            return {"position": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]}

        def set_prop_pose(self, prop, position, orientation):
            barrier.wait()
            return {"position": position, "orientation_wxyz": orientation}

        def park_props(self, names):
            barrier.wait()
            return {"poses": {}}

    async def main():
        world = IsaacWorld("sim-world")
        return await asyncio.gather(
            world.do_command({"command": "get_prop_pose", "prop": "a"}),
            world.do_command(
                {
                    "command": "set_prop_pose",
                    "prop": "b",
                    "position": [1, 2, 3],
                    "orientation_wxyz": [1, 0, 0, 0],
                }
            ),
            world.do_command({"command": "park_props", "props": ["c"]}),
        )

    previous = SimManager._instance
    SimManager._instance = BlockingSim()
    try:
        got, placed, parked = asyncio.run(main())
    finally:
        SimManager._instance = previous
    assert got["prop"] == "a"
    assert placed["position"] == [1, 2, 3]
    assert parked == {"poses": {}}


# --- lockstep `step` while paused ------------------------------------------


def _config(name, attrs):
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture
def world():
    previous = SimManager._instance
    manager = SimManager()
    SimManager._instance = manager
    thread = threading.Thread(target=manager.main_loop, daemon=True)
    thread.start()
    try:
        yield IsaacWorld.new(_config("sim-world", {"mock": True}), {})
    finally:
        manager.request_stop()
        thread.join(timeout=5)
        SimManager._instance = previous


def _do(world, command):
    return asyncio.run(world.do_command(command))


def test_step_requires_a_paused_world(world):
    with pytest.raises(RuntimeError, match="paused"):
        _do(world, {"command": "step"})
    _do(world, {"command": "pause"})
    for bad in (0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="ticks"):
            _do(world, {"command": "step", "ticks": bad})


def test_step_advances_exactly_ticks_of_rendering_dt(world):
    sim = SimManager.get()
    dt = sim.cfg.resolved_rendering_dt()
    _do(world, {"command": "pause"})
    before = sim.sim_time_s()
    time.sleep(0.05)
    assert sim.sim_time_s() == before  # paused: the clock holds
    result = _do(world, {"command": "step", "ticks": 3})
    assert result["ticks"] == 3
    assert result["sim_time_s"] == pytest.approx(before + 3 * dt)
    assert "episode_frames" not in result
    time.sleep(0.05)
    assert sim.sim_time_s() == pytest.approx(before + 3 * dt)
    _do(world, {"command": "play"})
    time.sleep(0.05)
    assert sim.sim_time_s() > before + 3 * dt


def test_replay_captures_one_frame_per_stepped_state(world, tmp_path):
    arm = IsaacArm.new(
        _config("arm", {"world": "sim-world", "asset": "lite6", "mock_dof": 6}), {}
    )
    IsaacGripper.new(
        _config("gripper", {"world": "sim-world", "arm": "arm", "asset": "lite6_gripper"}),
        {},
    )
    for name in ("scene-cam", "wrist-cam"):
        IsaacCamera.new(
            _config(name, {"world": "sim-world", "width": 32, "height": 24}), {}
        )
    states = [[0.1 * (i + 1)] * 6 for i in range(5)]

    async def replay():
        await world.do_command({"command": "pause"})
        await arm.do_command(
            {"command": "set_joint_positions_radians", "values": states[0]}
        )
        await world.do_command(
            {
                "command": "start_episode",
                "output_dir": str(tmp_path),
                "episode_id": "000001",
                "task": "replay",
                "arm": "arm",
                "gripper": "gripper",
                "fixed_camera": "scene-cam",
                "wrist_camera": "wrist-cam",
                "target": {
                    "prop_name": "block",
                    "position_m": [0.5, 0.0, 0.0],
                    "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        results = []
        for state in states[1:]:
            await arm.do_command(
                {"command": "set_joint_positions_radians", "values": state}
            )
            results.append(await world.do_command({"command": "step"}))
        await world.do_command({"command": "play"})
        await world.do_command({"command": "stop_episode", "outcome": "success"})
        return results

    results = asyncio.run(replay())
    # the start sample, then exactly one per stepped tick
    assert [r["episode_frames"] for r in results] == [2, 3, 4, 5]
    samples = pq.read_table(tmp_path / "000001" / "samples.parquet").to_pylist()
    assert len(samples) >= len(states)  # play adds samples after the replay
    for sample, state in zip(samples, states):
        # each state was applied before the tick that captured it
        assert sample["joint_position_rad"] == pytest.approx(state)


def test_step_paused_world_plays_the_timeline_for_one_update(monkeypatch):
    events = []
    timeline = SimpleNamespace(
        play=lambda: events.append("play"),
        pause=lambda: events.append("pause"),
        commit=lambda: events.append("commit"),
    )
    omni = SimpleNamespace(
        timeline=SimpleNamespace(get_timeline_interface=lambda: timeline)
    )
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.timeline", omni.timeline)
    manager = SimManager()
    manager.world = SimpleNamespace(step=lambda render: events.append(("step", render)))
    manager._step_paused_world()
    assert events == ["play", "commit", ("step", True), "pause", "commit"]
