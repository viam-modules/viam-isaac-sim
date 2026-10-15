import math

import pytest
from viam.proto.app.robot import ComponentConfig

from isaac_module.models.utils import (
    apply_camera_frame_to_attrs,
    apply_frame_to_attrs,
    frame_pose,
)
from isaac_module.spatial import quat_rotate
from isaac_module.sim_manager import (
    _parented_camera_pose,
    _reset_articulation_at_pose,
)


def test_no_frame():
    assert frame_pose(ComponentConfig(name="a")) == (None, None)


def test_translation_mm_to_meters():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.translation.x = 1000
    cfg.frame.translation.y = 2000
    cfg.frame.translation.z = 500
    pos, quat = frame_pose(cfg)
    assert pos == pytest.approx((1.0, 2.0, 0.5))
    assert quat is None  # no orientation set


def test_euler_yaw():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.orientation.euler_angles.yaw = math.pi / 2
    _, quat = frame_pose(cfg)
    assert quat_rotate(quat, (1, 0, 0)) == pytest.approx((0, 1, 0), abs=1e-9)


def test_quaternion_passthrough():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.orientation.quaternion.w = 1
    _, quat = frame_pose(cfg)
    assert quat == pytest.approx((1, 0, 0, 0))


def test_orientation_vector_degrees():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    ov = cfg.frame.orientation.vector_degrees
    ov.x, ov.y, ov.z, ov.theta = 0, 0, 1, 90
    _, quat = frame_pose(cfg)
    # OV (0,0,1) is identity direction; theta spins about it
    z = quat_rotate(quat, (0, 0, 1))
    assert z == pytest.approx((0, 0, 1), abs=1e-9)


def test_frame_wins_over_position_attr():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.translation.x = 3000
    attrs = apply_frame_to_attrs(cfg, {"position": [9, 9, 9]})
    assert attrs["position"] == [3.0, 0.0, 0.0]


def test_articulation_pose_becomes_reset_default_and_current_pose():
    events = []

    class FakeArticulation:
        def set_default_state(self, **pose):
            events.append(("default", pose))

        def set_world_pose(self, **pose):
            events.append(("current", pose))

    class FakeWorld:
        def reset(self):
            events.append(("reset", None))

    _reset_articulation_at_pose(
        FakeWorld(),
        FakeArticulation(),
        (-0.235, 0.0, 0.005),
        (1.0, 0.0, 0.0, 0.0),
    )

    expected_pose = {
        "position": [-0.235, 0.0, 0.005],
        "orientation": [1.0, 0.0, 0.0, 0.0],
    }
    assert events == [
        ("default", expected_pose),
        ("reset", None),
        ("current", expected_pose),
    ]


def test_arm_parented_camera_frame_becomes_ros_local_pose():
    cfg = ComponentConfig(name="wrist-cam")
    cfg.frame.parent = "pick-arm"
    cfg.frame.translation.x = 71.25
    cfg.frame.translation.y = -27.4
    cfg.frame.translation.z = 23.5
    ov = cfg.frame.orientation.vector_degrees
    ov.x, ov.y, ov.z, ov.theta = 0, 0, 1, 90
    attrs = apply_camera_frame_to_attrs(
        cfg,
        {
            "parent_prim": "/World/pick_arm/gripper/uflite_gripper_link",
            "local_position": [9, 9, 9],
            "local_orientation_rpy_deg": [180, -30, 0],
        },
    )

    position, orientation, axes = _parented_camera_pose(attrs)

    assert position == pytest.approx((0.07125, -0.0274, 0.0235))
    assert quat_rotate(orientation, (1, 0, 0)) == pytest.approx((0, 1, 0), abs=1e-9)
    assert axes == "ros"
    assert "local_orientation_rpy_deg" not in attrs
