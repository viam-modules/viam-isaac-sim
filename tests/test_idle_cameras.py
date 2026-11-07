import threading
from concurrent.futures import Future

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimManager, cameras_to_pause


def test_pauses_only_idle_unprotected_unpaused():
    last_reads = {"a": 0.0, "b": 9.5, "c": 0.0, "d": 0.0}
    assert cameras_to_pause(
        last_reads, paused={"c"}, protected={"d"}, now=10.0, idle_sec=2.0
    ) == ["a"]


def test_zero_idle_sec_never_pauses():
    assert cameras_to_pause({"a": 0.0}, set(), set(), now=100.0, idle_sec=0.0) == []


def test_world_validate_rejects_negative_idle_sec():
    for bad in (-1, 0.1):
        cfg = ComponentConfig(name="w", attributes=dict_to_struct({"camera_idle_sec": bad}))
        with pytest.raises(ValueError):
            IsaacWorld.validate_config(cfg)


@pytest.fixture
def mock_sim():
    from isaac_module.sim_manager import SimConfig

    previous = SimManager._instance
    manager = SimManager()
    SimManager._instance = manager
    t = threading.Thread(target=manager.main_loop, daemon=True)
    t.start()
    manager.ensure_booted(SimConfig(mock=True))
    yield manager
    manager.request_stop()
    t.join(timeout=5)
    SimManager._instance = previous


class FakeCam:
    def __init__(self):
        self.calls = []

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def get_rgba(self):
        import numpy as np

        return np.zeros((2, 2, 4), dtype=np.uint8)


def test_sim_config_defaults():
    from isaac_module.sim_manager import SimConfig

    cfg = SimConfig()
    assert cfg.camera_idle_sec == 0.0
    assert cfg.render_quality == "default"
    assert cfg.kit_settings == {}


def test_run_after_next_step_resolves_after_a_loop_without_blocking_drain(mock_sim):
    def task():
        fut = mock_sim.run_after_next_step(lambda: 42)
        assert not fut.done()
        return fut

    fut = mock_sim.run(task)
    assert isinstance(fut, Future)
    assert fut.result(timeout=5) == 42


def test_paused_camera_read_resumes_and_resolves_after_next_step(mock_sim):
    from isaac_module.sim_manager import IsaacCameraHandle

    cam = FakeCam()
    handle = IsaacCameraHandle(mock_sim, cam)
    handle.pause()
    rgb = handle.get_rgb()  # off the sim thread, as the camera model calls it
    assert rgb.shape == (2, 2, 3)
    assert cam.calls == ["pause", "resume"]
    assert handle.paused is False


def test_paused_camera_read_on_sim_thread_raises(mock_sim):
    from isaac_module.sim_manager import IsaacCameraHandle

    handle = IsaacCameraHandle(mock_sim, FakeCam())
    handle.pause()
    with pytest.raises(RuntimeError):
        mock_sim.run(handle.get_rgb)


def test_pause_tick_skips_episode_cameras(mock_sim):
    from isaac_module.sim_manager import IsaacCameraHandle, SimConfig

    recording, idle = (IsaacCameraHandle(mock_sim, FakeCam()) for _ in range(2))
    recording.last_read = idle.last_read = 0.0

    def tick():
        mock_sim.cfg = SimConfig(mock=True, camera_idle_sec=1.0)
        mock_sim._handles["rec"] = ({}, recording)
        mock_sim._handles["idle"] = ({}, idle)
        mock_sim._capture_handles = (recording,)
        try:
            mock_sim._pause_idle_cameras()
        finally:
            mock_sim._capture_handles = None
            del mock_sim._handles["rec"], mock_sim._handles["idle"]

    mock_sim.run(tick)
    assert idle.paused is True
    assert recording.paused is False


def test_paused_read_on_sim_thread_is_deferred_not_immediate(mock_sim):
    from isaac_module.sim_manager import IsaacCameraHandle

    handle = IsaacCameraHandle(mock_sim, FakeCam())
    handle.pause()

    def task():
        fut = handle._read(handle.capture_rgb)
        assert not fut.done()
        return fut

    assert mock_sim.run(task).result(timeout=5).shape == (2, 2, 3)


def test_two_reads_in_one_iteration_both_wait_for_the_step(mock_sim):
    from isaac_module.sim_manager import IsaacCameraHandle

    handle = IsaacCameraHandle(mock_sim, FakeCam())
    handle.pause()

    def task():
        futs = [handle._read(handle.capture_rgb) for _ in range(2)]
        assert not any(isinstance(f, Future) is False or f.done() for f in futs)
        return futs

    for fut in mock_sim.run(task):
        assert fut.result(timeout=5).shape == (2, 2, 3)


def test_sim_thread_read_refused_during_resume_window(mock_sim):
    from isaac_module.sim_manager import IsaacCameraHandle

    handle = IsaacCameraHandle(mock_sim, FakeCam())

    def task():
        handle.fresh_from = mock_sim._loop_index + 1
        handle._get(handle.capture_rgb)

    with pytest.raises(RuntimeError):
        mock_sim.run(task)


def test_resume_cameras_raises_on_sim_thread(mock_sim):
    from isaac_module.sim_manager import IsaacCameraHandle

    handle = IsaacCameraHandle(mock_sim, FakeCam())
    handle.pause()
    mock_sim._handles["x"] = ({}, handle)

    def task():
        # the mock path returns early; flip it on the sim thread because
        # main_loop reads self.mock every iteration and must not see False
        mock_sim.mock = False
        try:
            mock_sim._resume_cameras(["x"])
        finally:
            mock_sim.mock = True
            del mock_sim._handles["x"]

    with pytest.raises(RuntimeError, match="sim thread"):
        mock_sim.run(task)


def test_pause_tick_with_zero_idle_sec_pauses_nothing(mock_sim):
    from isaac_module.sim_manager import IsaacCameraHandle, SimConfig

    idle = IsaacCameraHandle(mock_sim, FakeCam())
    idle.last_read = 0.0

    def tick():
        mock_sim.cfg = SimConfig(mock=True, camera_idle_sec=0.0)
        mock_sim._handles["idle"] = ({}, idle)
        try:
            mock_sim._pause_idle_cameras()
        finally:
            del mock_sim._handles["idle"]

    mock_sim.run(tick)
    assert idle.paused is False
