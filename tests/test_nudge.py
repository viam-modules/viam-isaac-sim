import math
import threading
from types import SimpleNamespace

import pytest

from isaac_module.nudge import NudgeConfig, NudgeSchedule
from isaac_module.sim_manager import (
    ArmHandle,
    CameraHandle,
    GripperHandle,
    IsaacArmHandle,
    JointTrajectoryPoint,
    SimConfig,
    SimManager,
)

DT = 1.0 / 60.0


def _offsets(seed, steps=600, config=None):
    schedule = NudgeSchedule(config or NudgeConfig(2.0, 1.0, 0.5), seed, 6)
    return [schedule.step(DT) for _ in range(steps)]


def test_schedule_is_seeded():
    assert _offsets(7) == _offsets(7)
    assert _offsets(7) != _offsets(8)


def test_nudges_stay_within_amplitude_and_fade_back_to_zero():
    offsets = _offsets(3)
    limit = math.radians(2.0)
    assert all(abs(value) <= limit for step in offsets for value in step)
    active = [any(step) for step in offsets]
    assert any(active) and not all(active)
    assert not active[0]
    # Every nudge ends by returning to exactly zero.
    ends = [i for i in range(1, len(active)) if active[i - 1] and not active[i]]
    assert ends
    assert all(offsets[i] == [0.0] * 6 for i in ends)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude_deg": 0.0, "interval_s": 1.0},
        {"amplitude_deg": 1.0, "interval_s": -1.0},
        {"amplitude_deg": 1.0, "interval_s": 1.0, "duration_s": math.nan},
        {"amplitude_deg": [0.0] * 6, "interval_s": 1.0},
        {"amplitude_deg": [1.0, -1.0, 1.0, 1.0, 1.0, 1.0], "interval_s": 1.0},
    ],
)
def test_config_rejects_non_positive_values(kwargs):
    with pytest.raises(ValueError):
        NudgeConfig(**kwargs)


def test_per_joint_amplitudes_bound_each_joint():
    amplitudes = [2.5, 3.3, 2.9, 0.0, 3.2, 5.0]
    offsets = _offsets(3, config=NudgeConfig(amplitudes, 1.0, 0.5))
    for joint, amplitude in enumerate(amplitudes):
        values = [abs(step[joint]) for step in offsets]
        assert max(values) <= math.radians(amplitude)
    assert max(abs(step[5]) for step in offsets) > math.radians(2.5)
    assert all(step[3] == 0.0 for step in offsets)


def test_per_joint_amplitudes_must_match_joint_count():
    with pytest.raises(ValueError, match="6 values for 7 joints"):
        NudgeSchedule(NudgeConfig([1.0] * 6, 1.0), 0, 7)


class FakeArticulation:
    dof_names = tuple(f"joint{index}" for index in range(1, 7))

    def __init__(self):
        self.applied = []

    def apply_action(self, action):
        self.applied.append([float(value) for value in action["joint_positions"]])


def _isaac_arm():
    sim = SimpleNamespace(
        run=lambda fn: fn(),
        _isaac=SimpleNamespace(ArticulationAction=lambda **kwargs: kwargs),
    )
    articulation = FakeArticulation()
    return IsaacArmHandle(sim, articulation, None, "/World/arm"), articulation


def test_isaac_arm_drives_target_plus_nudge_but_keeps_clean_target():
    arm, articulation = _isaac_arm()
    arm.set_joint_targets([1.0] * 6)
    arm.set_nudge([0.1] * 6)
    assert articulation.applied[-1] == pytest.approx([1.1] * 6)
    assert arm._target == [1.0] * 6
    assert arm.nudge_offsets() == [0.1] * 6

    arm.set_joint_targets([2.0] * 6)
    assert articulation.applied[-1] == pytest.approx([2.1] * 6)
    assert arm._target == [2.0] * 6

    arm.set_nudge([0.0] * 6)
    assert articulation.applied[-1] == [2.0] * 6
    assert arm.nudge_offsets() is None


def test_isaac_arm_trajectory_steps_include_nudge_but_record_clean_target():
    callbacks = {}
    world = SimpleNamespace(
        add_physics_callback=lambda name, fn: callbacks.__setitem__(name, fn),
        remove_physics_callback=lambda name: callbacks.pop(name),
    )
    sim = SimpleNamespace(
        run=lambda fn: fn(),
        world=world,
        _isaac=SimpleNamespace(ArticulationAction=lambda **kwargs: kwargs),
    )
    articulation = FakeArticulation()
    arm = IsaacArmHandle(sim, articulation, None, "/World/arm")
    arm.start_trajectory(
        [
            JointTrajectoryPoint(0.0, (1.0,) * 6),
            JointTrajectoryPoint(0.05, (2.0,) * 6),
        ]
    )
    (step,) = callbacks.values()

    step(0.05)
    assert articulation.applied[-1] == [1.0] * 6
    arm.set_nudge([0.1] * 6)
    step(0.05)
    assert articulation.applied[-1] == pytest.approx([2.1] * 6)
    assert arm._target == [2.0] * 6


class FakePhysicsWorld:
    def __init__(self):
        self.callbacks = {}
        self.current_time = 0.0

    def add_physics_callback(self, name, callback):
        self.callbacks[name] = callback

    def remove_physics_callback(self, name):
        self.callbacks.pop(name)


class NudgedArm(ArmHandle):
    def __init__(self):
        self.nudge = None
        self.history = []

    def joint_names(self):
        return [f"joint{index}" for index in range(1, 7)]

    def set_nudge(self, offsets):
        self.nudge = list(offsets) if any(offsets) else None
        self.history.append(list(offsets))

    def nudge_offsets(self):
        return self.nudge

    def capture_state(self):
        return [0.0] * 6, [0.0] * 6, [0.5] * 6, [0.0] * 3, [1.0, 0.0, 0.0, 0.0]


def test_episode_nudges_arm_and_records_clean_target_with_nudge(tmp_path, monkeypatch):
    class FakeCamera(CameraHandle):
        def capture_rgb(self):
            return None

    class FakeGripper(GripperHandle):
        def capture_state(self):
            return 0.0, 0.0

    samples = []

    class FakeEpisodeWriter:
        failed = False

        def __init__(self, output_dir, *unused_args):
            self.partial_dir = tmp_path / ".000042.partial"
            self.sample_count = 0

        def submit(self, sample, *unused_frames):
            samples.append(sample)
            self.sample_count += 1

        def close(self, outcome):
            return tmp_path / "000042"

        def stage_start_frames(self):
            return (0,)

    monkeypatch.setattr("isaac_module.episode_capture.EpisodeWriter", FakeEpisodeWriter)
    manager = SimManager()
    manager.cfg = SimConfig()
    manager._sim_thread_id = threading.get_ident()
    manager.world = world = FakePhysicsWorld()
    arm = NudgedArm()
    manager._handles.update(
        {
            "pick-arm": ({"asset": "arm"}, arm),
            "pick-gripper": ({"asset": "gripper"}, FakeGripper()),
            "scene-cam": ({}, FakeCamera()),
            "wrist-cam": ({}, FakeCamera()),
        }
    )

    manager.start_episode(
        str(tmp_path),
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
        nudge=NudgeConfig(amplitude_deg=2.0, interval_s=0.05, duration_s=1.0),
    )
    for _ in range(12):
        world.callbacks["episode_nudge"](0.05)
        world.callbacks["episode_capture"](0.05)

    expected = NudgeSchedule(NudgeConfig(2.0, 0.05, 1.0), 42, 6)
    assert arm.history == [expected.step(0.05) for _ in range(12)]
    nudged = [sample for sample in samples if any(sample.joint_target_nudge_rad)]
    assert nudged
    assert all(sample.joint_target_rad == (0.5,) * 6 for sample in samples)

    manager.stop_episode("success")
    assert "episode_nudge" not in world.callbacks
    assert arm.history[-1] == [0.0] * 6


def test_nudges_are_rejected_in_mock_mode(tmp_path):
    manager = SimManager()
    manager.mock = True
    manager.cfg = SimConfig(mock=True)
    with pytest.raises(RuntimeError, match="unavailable in mock mode"):
        manager.start_episode(
            str(tmp_path),
            "000042",
            42,
            "Pick up the block",
            "pick-arm",
            "pick-gripper",
            "scene-cam",
            "wrist-cam",
            {},
            nudge=NudgeConfig(2.0, 1.0),
        )


def test_isaac_arm_stop_mid_nudge_keeps_clean_target():
    class StoppableArticulation(FakeArticulation):
        def get_joint_positions(self, joint_indices=None):
            return [1.1] * 6

        def get_joint_velocities(self, joint_indices=None):
            return [0.0] * 6

        def set_joint_velocities(self, velocities, joint_indices=None):
            pass

    sim = SimpleNamespace(
        run=lambda fn: fn(),
        _step_callbacks={},
        _isaac=SimpleNamespace(ArticulationAction=lambda **kwargs: kwargs),
    )
    articulation = StoppableArticulation()
    arm = IsaacArmHandle(sim, articulation, None, "/World/arm")
    arm.set_joint_targets([1.0] * 6)
    arm.set_nudge([0.1] * 6)

    arm.stop_trajectory()
    assert arm._target == pytest.approx([1.0] * 6)
    assert articulation.applied[-1] == pytest.approx([1.1] * 6)
