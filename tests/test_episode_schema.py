import json

import pytest

from isaac_module.episode_schema import (
    CameraStreamV2,
    EpisodeManifestV3,
    EpisodeSampleV2,
    EpisodeStageV3,
    RobotDescriptorV2,
    TargetPoseV2,
)

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _sample(**changes) -> EpisodeSampleV2:
    values = {
        "frame_index": 0,
        "sim_time_s": 1.25,
        "joint_position_rad": (0.0,) * 6,
        "joint_velocity_rad_s": (0.0,) * 6,
        "joint_target_rad": (0.1,) * 6,
        "ee_position_m": (0.5, 0.0, 0.3),
        "ee_rotation_wxyz": (1.0, 0.0, 0.0, 0.0),
        "gripper_command": 1.0,
        "gripper_state": 1.0,
    }
    values.update(changes)
    return EpisodeSampleV2(**values)


def _manifest(**changes) -> EpisodeManifestV3:
    values = {
        "episode_id": "000042",
        "seed": 42,
        "started_at": "2026-09-16T12:00:00Z",
        "robot": RobotDescriptorV2(
            arm_component="pick-arm",
            arm_model="lite6",
            joint_names=JOINT_NAMES,
            gripper_component="pick-gripper",
            gripper_model="lite6_gripper",
            gripper_mode="binary",
        ),
        "cameras": (
            CameraStreamV2("fixed", "scene-cam", "fixed.mp4", 640, 360),
            CameraStreamV2("wrist", "wrist-cam", "wrist.mp4", 640, 360),
        ),
        "target": TargetPoseV2("pick-block", (0.6, 0.1, 0.015), (1.0, 0.0, 0.0, 0.0)),
        "task": "Pick up the block",
        "stages": (
            EpisodeStageV3(0, 40, "Center the gripper above the block"),
            EpisodeStageV3(40, 120, "Grasp and lift the block"),
        ),
        "outcome": "success",
        "sample_count": 120,
        "sample_rate_hz": 15.0,
    }
    values.update(changes)
    return EpisodeManifestV3(**values)


def test_valid_episode_contract_is_json_native():
    payload = json.loads(_manifest().to_json())

    assert payload["schema_version"] == 3
    assert payload["sample_rate_hz"] == 15.0
    assert payload["robot"]["joint_names"] == list(JOINT_NAMES)
    assert {camera["role"] for camera in payload["cameras"]} == {"fixed", "wrist"}
    assert _sample().joint_target_rad == (0.1,) * 6
    assert payload["stages"][0] == {
        "start_frame": 0,
        "end_frame_exclusive": 40,
        "task": "Center the gripper above the block",
    }


def test_manifest_accepts_optional_second_fixed_camera():
    manifest = _manifest(
        cameras=(
            CameraStreamV2("fixed", "scene-a", "fixed.mp4", 640, 360),
            CameraStreamV2("wrist", "wrist", "wrist.mp4", 640, 360),
            CameraStreamV2("fixed_2", "scene-b", "fixed_2.mp4", 640, 360),
        )
    )

    assert [camera.role for camera in manifest.cameras] == [
        "fixed",
        "wrist",
        "fixed_2",
    ]


def test_sample_rejects_wrong_joint_arity():
    with pytest.raises(ValueError, match="joint_position_rad must have 6"):
        _sample(joint_position_rad=(0.0,) * 7)


def test_sample_rejects_gripper_outside_normalized_range():
    with pytest.raises(ValueError, match="gripper_command"):
        _sample(gripper_command=1.01)


def test_sample_rejects_non_unit_orientation():
    with pytest.raises(ValueError, match="unit quaternion"):
        _sample(ee_rotation_wxyz=(0.0, 0.0, 0.0, 0.0))


def test_manifest_rejects_missing_wrist_camera():
    fixed = CameraStreamV2("fixed", "scene-a", "a.mp4", 640, 360)
    with pytest.raises(ValueError, match="one fixed and one wrist"):
        _manifest(
            cameras=(fixed, CameraStreamV2("fixed", "scene-b", "b.mp4", 640, 360))
        )


def test_manifest_accepts_positive_finite_sample_rate_and_rejects_invalid_values():
    assert _manifest(sample_rate_hz=30).sample_rate_hz == 30.0
    for value in (0, -1, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            _manifest(sample_rate_hz=value)


def test_manifest_rejects_invalid_outcome():
    with pytest.raises(ValueError, match="outcome"):
        _manifest(outcome="unknown")


def test_manifest_rejects_naive_started_at():
    with pytest.raises(ValueError, match="timezone"):
        _manifest(started_at="2026-09-16T12:00:00")


def test_manifest_rejects_noncontiguous_or_incomplete_stages():
    with pytest.raises(ValueError, match="contiguous"):
        _manifest(
            stages=(
                EpisodeStageV3(0, 40, "Center"),
                EpisodeStageV3(41, 120, "Lift"),
            )
        )
    with pytest.raises(ValueError, match="sample_count"):
        _manifest(stages=(EpisodeStageV3(0, 119, "Pick"),))


def test_stage_requires_enough_frames_for_a_32_action_window():
    with pytest.raises(ValueError, match="at least 33 frames"):
        EpisodeStageV3(0, 32, "Rotate")
    assert EpisodeStageV3(0, 33, "Rotate").end_frame_exclusive == 33
