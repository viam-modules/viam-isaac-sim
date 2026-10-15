import math
import threading

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.camera import IsaacCamera
from isaac_module.sim_manager import (
    SimManager,
    _apply_camera_configuration,
)


def _config(name, attrs):
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _camera_manager():
    manager = SimManager()
    manager.mock = True
    manager._booted.set()
    manager._sim_thread_id = threading.get_ident()
    return manager


def test_camera_frame_rejects_legacy_pose_conflicts():
    config = _config(
        "scene-cam",
        {"world": "sim-world", "position": [1, 2, 3]},
    )
    config.frame.parent = "world"

    with pytest.raises(ValueError, match="attributes.position conflicts"):
        IsaacCamera.validate_config(config)

    config = _config(
        "scene-cam",
        {"world": "sim-world", "target": [0, 0, 0]},
    )
    config.frame.parent = "world"
    config.frame.orientation.quaternion.w = 1

    with pytest.raises(ValueError, match="attributes.target conflicts"):
        IsaacCamera.validate_config(config)


def test_parented_camera_frame_rejects_legacy_local_pose():
    config = _config(
        "wrist-cam",
        {
            "world": "sim-world",
            "parent_prim": "/World/arm/wrist",
            "local_position": [0, 0, 0.05],
        },
    )
    config.frame.parent = "pick-arm"

    with pytest.raises(ValueError, match="attributes.local_position conflicts"):
        IsaacCamera.validate_config(config)


def test_camera_live_configuration_reuses_handle_and_updates_cache():
    manager = _camera_manager()
    initial = {
        "world": "sim-world",
        "position": [0.0, -0.5, 0.2],
        "target": [0.0, 0.0, 0.0],
        "width": 640,
        "height": 480,
    }
    handle = manager.create_camera("scene-cam", initial)

    updated = {
        **initial,
        "position": [0.1, -0.6, 0.25],
        "fov_deg": 65.0,
        "near_clip_m": 0.02,
    }
    reconfigured = manager.create_camera("scene-cam", updated)

    assert reconfigured is handle
    assert manager._handles["scene-cam"][0] == updated


def test_camera_reconfigure_rejects_restart_fields_and_active_capture():
    manager = _camera_manager()
    initial = {
        "world": "sim-world",
        "position": [0.0, -0.5, 0.2],
        "width": 640,
        "height": 480,
    }
    manager.create_camera("scene-cam", initial)

    with pytest.raises(ValueError, match="module restart.*width"):
        manager.create_camera("scene-cam", {**initial, "width": 1280})

    manager._episode_id = "000001"
    with pytest.raises(RuntimeError, match="during episode capture"):
        manager.create_camera(
            "scene-cam", {**initial, "position": [0.0, -0.6, 0.2]}
        )


class _FakeIsaacCamera:
    def __init__(self):
        self.position = (0.0, 0.0, 0.0)
        self.orientation = (1.0, 0.0, 0.0, 0.0)
        self.near = 0.01
        self.far = 1000.0
        self.aperture = 20.0
        self.focal_length = 0.0

    def set_world_pose(self, position=None, orientation=None, camera_axes="world"):
        if position is not None:
            self.position = tuple(position)
        if orientation is not None:
            self.orientation = tuple(orientation)

    def get_world_pose(self, camera_axes="world"):
        return self.position, self.orientation

    def set_clipping_range(self, near_distance=None, far_distance=None):
        if near_distance is not None:
            self.near = near_distance
        if far_distance is not None:
            self.far = far_distance

    def get_clipping_range(self):
        return self.near, self.far

    def get_horizontal_aperture(self):
        return self.aperture

    def set_focal_length(self, value):
        self.focal_length = value

    def get_focal_length(self):
        return self.focal_length


def test_isaac_camera_configuration_applies_and_reads_back_state():
    camera = _FakeIsaacCamera()

    _apply_camera_configuration(
        camera,
        {
            "position": [0.0, -0.5, 0.2],
            "target": [0.0, 0.0, 0.0],
            "near_clip_m": 0.02,
            "fov_deg": 60.0,
        },
    )

    assert camera.position == pytest.approx((0.0, -0.5, 0.2))
    assert camera.near == pytest.approx(0.02)
    assert camera.focal_length == pytest.approx(
        camera.aperture / (2.0 * math.tan(math.radians(60.0) / 2.0))
    )
