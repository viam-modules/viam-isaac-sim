import asyncio
import struct
import threading

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.camera import IsaacCamera
from isaac_module.models.utils import apply_camera_frame_to_attrs, get_attrs
from isaac_module.sim_manager import SimManager, _apply_camera_configuration, camera_axes
from test_camera_reconfigure import _FakeIsaacCamera
from test_pointcloud import parse_pcd


def _config(name, attrs):
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture
def manager(monkeypatch):
    m = SimManager()
    m.mock = True
    m._booted.set()
    m._sim_thread_id = threading.get_ident()
    monkeypatch.setattr(SimManager, "_instance", m)
    return m


def test_fixed_camera_frame_orientation_uses_ros_axes():
    cfg = _config("cam", {"world": "sim-world"})
    cfg.frame.parent = "world"
    cfg.frame.orientation.vector_degrees.x = 1.0
    attrs = apply_camera_frame_to_attrs(cfg, get_attrs(cfg))
    assert attrs["camera_axes"] == "ros"
    assert camera_axes(attrs) == "ros"


def _identity_frame_cfg(**attrs):
    cfg = _config("cam", {"world": "sim-world", **attrs})
    cfg.frame.parent = "world"
    cfg.frame.translation.x = 1000
    cfg.frame.orientation.quaternion.w = 1  # viam-server's default
    return cfg


def test_identity_frame_orientation_with_target_validates():
    IsaacCamera.validate_config(_identity_frame_cfg(target=[0, 0, 0]))


def test_identity_frame_orientation_is_unset_for_attrs():
    cfg = _identity_frame_cfg()
    attrs = apply_camera_frame_to_attrs(cfg, get_attrs(cfg))
    assert attrs["position"] == [1.0, 0.0, 0.0]
    assert "orientation_wxyz" not in attrs
    assert "camera_axes" not in attrs


def test_non_identity_frame_orientation_with_target_rejected():
    cfg = _identity_frame_cfg(target=[0, 0, 0])
    cfg.frame.orientation.vector_degrees.x = 1.0
    cfg.frame.orientation.vector_degrees.theta = -90
    with pytest.raises(ValueError, match="target conflicts"):
        IsaacCamera.validate_config(cfg)


def test_non_identity_frame_orientation_sets_ros_axes():
    cfg = _identity_frame_cfg()
    cfg.frame.orientation.vector_degrees.x = 1.0
    cfg.frame.orientation.vector_degrees.theta = -90
    attrs = apply_camera_frame_to_attrs(cfg, get_attrs(cfg))
    assert attrs["camera_axes"] == "ros"
    assert "orientation_wxyz" in attrs


def test_target_and_position_only_cameras_keep_world_axes():
    cfg = _config("cam", {"world": "sim-world", "target": [0, 0, 0]})
    cfg.frame.parent = "world"
    cfg.frame.translation.x = 1000
    attrs = apply_camera_frame_to_attrs(cfg, get_attrs(cfg))
    assert "camera_axes" not in attrs
    assert camera_axes(attrs) == "world"


def test_parented_axes_rules():
    assert camera_axes({"parent_prim": "/p", "local_orientation_wxyz": [1, 0, 0, 0]}) == "ros"
    assert camera_axes({"parent_prim": "/p"}) == "usd"
    assert (
        camera_axes(
            {"parent_prim": "/p", "local_orientation_wxyz": [1, 0, 0, 0], "local_camera_axes": "world"}
        )
        == "world"
    )


class _AxesRecordingCamera(_FakeIsaacCamera):
    # _apply_camera_configuration also touches clipping/focal methods, which
    # the existing fake provides
    axes = None

    def set_world_pose(self, position=None, orientation=None, camera_axes="world"):
        self.axes = camera_axes
        super().set_world_pose(position, orientation, camera_axes)


def test_apply_camera_configuration_poses_with_ros_axes():
    cam = _AxesRecordingCamera()
    _apply_camera_configuration(
        cam, {"position": [1, 2, 3], "orientation_wxyz": [1, 0, 0, 0], "camera_axes": "ros"}
    )
    assert cam.axes == "ros"


def test_mock_handle_rgbd_and_intrinsics(manager):
    h = manager.create_camera("cam", {"world": "sim-world", "depth": True, "width": 64, "height": 48})
    rgb, depth = h.get_rgbd()
    assert rgb.shape == (48, 64, 3) and depth.shape == (48, 64)
    assert depth.dtype == np.float32
    assert float(depth.min()) == pytest.approx(1.0) and float(depth.max()) == pytest.approx(2.0)
    fx, fy, cx, cy = h.intrinsics()
    assert fx == pytest.approx(32.0 / np.tan(np.radians(35.0)))
    assert (fy, cx, cy) == (pytest.approx(fx), 32.0, 24.0)


def test_depth_change_requires_restart(manager):
    manager.create_camera("cam", {"world": "sim-world", "depth": False})
    with pytest.raises(ValueError, match="module restart.*depth"):
        manager.create_camera("cam", {"world": "sim-world", "depth": True})


def _depth_cam(manager, **extra):
    attrs = {"world": "sim-world", "depth": True, "width": 64, "height": 48, **extra}
    return IsaacCamera.new(_config("dcam", attrs), {})


def test_validate_max_depth_requires_depth():
    with pytest.raises(ValueError, match="max_depth_m"):
        IsaacCamera.validate_config(_config("c", {"world": "w", "max_depth_m": 2.0}))
    with pytest.raises(ValueError, match="max_depth_m"):
        IsaacCamera.validate_config(_config("c", {"world": "w", "depth": True, "max_depth_m": 0}))


def test_get_images_returns_color_and_depth(manager):
    cam = _depth_cam(manager)
    images, _ = asyncio.run(cam.get_images())
    assert [i.name for i in images] == ["color", "depth"]
    dep = images[1]
    assert dep.mime_type == "image/vnd.viam.dep"
    assert dep.data[:8] == b"DEPTHMAP"
    assert struct.unpack(">qq", dep.data[8:24]) == (64, 48)
    assert len(dep.data) == 24 + 64 * 48 * 2


def test_point_cloud_is_in_placement_axes(manager):
    cam = _depth_cam(manager)  # no frame -> world axes (X forward)
    data, mime = asyncio.run(cam.get_point_cloud())
    assert mime == "pointcloud/pcd"
    header, arr = parse_pcd(data)
    assert header["FIELDS"] == "x y z rgb"
    assert int(header["POINTS"]) == 64 * 48
    assert arr["x"].min() >= 1.0 - 1e-5 and arr["x"].max() <= 2.0 + 1e-5


def test_max_depth_filters_cloud_and_is_viam_side(manager):
    cam = _depth_cam(manager, max_depth_m=1.5)
    _, arr = parse_pcd(asyncio.run(cam.get_point_cloud())[0])
    assert 0 < len(arr) < 64 * 48
    assert "max_depth_m" not in manager._handles["dcam"][0]


def test_point_cloud_without_depth_raises(manager):
    cam = IsaacCamera.new(_config("rgb", {"world": "sim-world"}), {})
    with pytest.raises(NotImplementedError, match='"depth": true'):
        asyncio.run(cam.get_point_cloud())


def test_properties_report_intrinsics(manager):
    cam = _depth_cam(manager)
    props = asyncio.run(cam.get_properties())
    assert props.supports_pcd is True
    assert props.intrinsic_parameters.width_px == 64
    assert props.intrinsic_parameters.center_x_px == pytest.approx(32.0)


def test_max_depth_applies_live(manager):
    cam = _depth_cam(manager, max_depth_m=1.5)
    cam.reconfigure(
        _config("dcam", {"world": "sim-world", "depth": True, "width": 64, "height": 48}), {}
    )
    _, arr = parse_pcd(asyncio.run(cam.get_point_cloud())[0])
    assert len(arr) == 64 * 48


def test_get_images_honors_filter_source_names(manager):
    cam = _depth_cam(manager)
    only_depth, _ = asyncio.run(cam.get_images(filter_source_names=["depth"]))
    assert [i.name for i in only_depth] == ["depth"]
    only_color, _ = asyncio.run(cam.get_images(filter_source_names=["color"]))
    assert [i.name for i in only_color] == ["color"]


def test_rgb_only_camera_keeps_component_name(manager):
    cam = IsaacCamera.new(_config("rgbcam", {"world": "sim-world", "width": 64, "height": 48}), {})
    images, _ = asyncio.run(cam.get_images())
    assert [i.name for i in images] == ["rgbcam"]
    assert asyncio.run(cam.get_images(filter_source_names=["color"]))[0] == []


def test_depth_first_orders_depth_before_color(manager):
    """Consumers that decode only the first image (obstacles-depth) need it."""
    cam = _depth_cam(manager, depth_first=True)
    images, _ = asyncio.run(cam.get_images())
    assert [i.name for i in images] == ["depth", "color"]
    with pytest.raises(ValueError, match="depth_first"):
        IsaacCamera.validate_config(
            _config("c", {"world": "sim-world", "depth_first": True})
        )
