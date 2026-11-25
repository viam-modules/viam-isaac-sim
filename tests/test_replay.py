"""Replaying recorded states: world DoCommands off the event loop, and a
lockstep `step` while paused."""

import asyncio
import dataclasses
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
from isaac_module.sim_manager import SimConfig, SimManager


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


class _TimelineWorld:
    """A stand-in Isaac World: time advances only while the timeline plays."""

    def __init__(self, dt):
        self.dt = dt
        self.current_time = 0.0
        self.playing = False
        self.steps = 0

    def is_playing(self):
        return self.playing

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def step(self, render=True):
        self.steps += 1
        if self.playing:
            self.current_time += self.dt


@pytest.fixture
def real_time_sim(monkeypatch):
    """The non-mock loop with real_time pacing, over a stand-in World."""
    dt = 0.4  # long, so a command left waiting out the period shows
    fake = _TimelineWorld(dt)
    timeline = SimpleNamespace(play=fake.play, pause=fake.pause, commit=lambda: None)
    omni = SimpleNamespace(
        timeline=SimpleNamespace(get_timeline_interface=lambda: timeline)
    )
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.timeline", omni.timeline)
    manager = SimManager()
    manager.cfg = dataclasses.replace(
        SimConfig(), physics_dt=dt, rendering_dt=dt, real_time=True
    )

    def boot():
        manager.world = fake

    manager._boot = boot
    manager._boot_requested.set()
    thread = threading.Thread(target=manager.main_loop, daemon=True)
    thread.start()
    assert manager._booted.wait(5)
    yield manager, fake
    manager.request_stop()
    thread.join(timeout=5)


def test_real_step_plays_the_timeline_for_exactly_its_ticks(real_time_sim):
    manager, fake = real_time_sim
    assert manager.step(1)["sim_time_s"] == pytest.approx(fake.dt)
    assert manager.step(3) == {"sim_time_s": pytest.approx(4 * fake.dt), "ticks": 3}
    assert not fake.playing
    time.sleep(0.1)
    assert fake.current_time == pytest.approx(4 * fake.dt)


def test_paused_real_time_loop_serves_commands_without_waiting_out_the_period(
    real_time_sim,
):
    manager, fake = real_time_sim
    waits = []
    for _ in range(5):
        time.sleep(0.05)
        start = time.perf_counter()
        manager.run(lambda: None)
        waits.append(time.perf_counter() - start)
    # paced like a playing world, each would wait up to 0.4 s
    assert max(waits) < 0.1


def test_paused_loop_serves_commands_between_renders(real_time_sim):
    # a paused frame between every command and the step after it would make
    # each replayed state wait out an extra render
    manager, fake = real_time_sim
    time.sleep(0.05)
    before = fake.steps
    for _ in range(10):
        manager.run(lambda: None)
    assert fake.steps - before <= 2  # renders hold their own cadence
