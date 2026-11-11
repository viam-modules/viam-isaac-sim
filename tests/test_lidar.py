import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.lidar import (
    LIDAR_PRESETS,
    IsaacLidarHandle,
    MockLidarHandle,
    _read_lidar_scan,
    effective_rate,
)
from isaac_module.models.lidar import IsaacLidar
from isaac_module.models.utils import apply_lidar_frame_to_attrs, get_attrs
from isaac_module.sim_manager import SimManager
from test_pointcloud import parse_pcd


@pytest.fixture
def manager(monkeypatch):
    m = SimManager()
    m.mock = True
    m._booted.set()
    m._sim_thread_id = threading.get_ident()
    monkeypatch.setattr(SimManager, "_instance", m)
    return m


def test_presets():
    assert {"rplidar_s2e", "ouster_os1_32", "robosense_vega"} <= set(LIDAR_PRESETS)
    assert LIDAR_PRESETS["rplidar_s2e"].planar
    assert LIDAR_PRESETS["robosense_vega"].stand_in_for == "ouster_os1_32"
    assert LIDAR_PRESETS["robosense_vega"].configs == LIDAR_PRESETS["ouster_os1_32"].configs


def test_effective_rate():
    assert effective_rate({"sensor": "rplidar_s2e"}) == LIDAR_PRESETS["rplidar_s2e"].native_hz
    assert effective_rate({"sensor": "rplidar_s2e", "rate_hz": 2}) == 2.0
    with pytest.raises(ValueError, match="native rate"):
        effective_rate({"sensor": "rplidar_s2e", "rate_hz": 1000})


def test_mock_no_scan_until_one_rotation(manager):
    h = MockLidarHandle(manager, {"sensor": "rplidar_s2e"})
    assert h.latest_scan() is None
    manager._mock_sim_time_s = 0.1
    stamp, points = h.latest_scan()
    assert stamp == pytest.approx(0.1)
    assert points.shape == (360, 3)


def test_mock_3d_has_rings(manager):
    manager._mock_sim_time_s = 1.0
    _, points = MockLidarHandle(manager, {"sensor": "ouster_os1_32"}).latest_scan()
    assert points.shape == (360 * 16, 3)
    assert np.ptp(points[:, 2]) > 0


def test_mock_decimation(manager):
    h = MockLidarHandle(manager, {"sensor": "rplidar_s2e", "rate_hz": 2})
    manager._mock_sim_time_s = 0.1
    assert h.latest_scan()[0] == pytest.approx(0.1)
    manager._mock_sim_time_s = 0.3
    assert h.latest_scan()[0] == pytest.approx(0.1)  # < 0.5 s since last
    manager._mock_sim_time_s = 0.6
    assert h.latest_scan()[0] == pytest.approx(0.6)


def test_mock_reconfigure_rate(manager):
    h = MockLidarHandle(manager, {"sensor": "rplidar_s2e", "rate_hz": 2})
    h.reconfigure({"sensor": "rplidar_s2e", "rate_hz": 10})
    manager._mock_sim_time_s = 0.1
    h.latest_scan()
    manager._mock_sim_time_s = 0.2
    assert h.latest_scan()[0] == pytest.approx(0.2)


def test_create_lidar_caches_and_reconfigures(manager):
    attrs = {"world": "sim-world", "sensor": "rplidar_s2e", "rate_hz": 2}
    h = manager.create_lidar("lid", attrs)
    assert manager.create_lidar("lid", dict(attrs)) is h
    assert manager.create_lidar("lid", {**attrs, "rate_hz": 5, "position": [1, 0, 0]}) is h


def test_create_lidar_restart_only_fields(manager):
    attrs = {"world": "sim-world", "sensor": "rplidar_s2e"}
    manager.create_lidar("lid", attrs)
    with pytest.raises(ValueError, match="module restart.*sensor"):
        manager.create_lidar("lid", {**attrs, "sensor": "ouster_os1_32"})


def test_closed_lidar_can_come_back_with_a_new_sensor(manager):
    lid = IsaacLidar.new(_config("lid", {"world": "sim-world", "sensor": "rplidar_s2e"}), {})
    old = lid._handle
    asyncio.run(lid.close())
    assert "lid" not in manager._handles
    new = manager.create_lidar("lid", {"world": "sim-world", "sensor": "ouster_os1_32"})
    assert new is not old


def test_remove_lidar_tears_down_on_the_sim_thread(manager):
    closed = []
    handle = SimpleNamespace(close=lambda: closed.append(True))
    manager.mock = False
    manager._handles["lid"] = ({}, handle)
    manager._render_callbacks["lidar:lid"] = lambda: None
    manager.remove_lidar("lid")
    assert closed == [True] and "lidar:lid" not in manager._render_callbacks


def test_create_lidar_rejects_rate_above_native_on_reconfigure(manager):
    attrs = {"world": "sim-world", "sensor": "rplidar_s2e"}
    manager.create_lidar("lid", attrs)
    with pytest.raises(ValueError, match="native rate"):
        manager.create_lidar("lid", {**attrs, "rate_hz": 99})


def test_mock_scan_refreshes_after_sim_time_reset(manager):
    h = MockLidarHandle(manager, {"sensor": "rplidar_s2e", "rate_hz": 2})
    manager._mock_sim_time_s = 0.6
    assert h.latest_scan()[0] == pytest.approx(0.6)
    manager._mock_sim_time_s = 0.2
    assert h.latest_scan()[0] == pytest.approx(0.2)


def _config(name, attrs):
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.mark.parametrize(
    "attrs, match",
    [
        ({"world": "w"}, "sensor"),
        ({"world": "w", "sensor": "nope"}, "rplidar_s2e"),
        ({"world": "w", "sensor": "rplidar_s2e", "rate_hz": 0}, "rate_hz"),
        ({"world": "w", "sensor": "rplidar_s2e", "rate_hz": 99}, "native rate"),
        ({"world": "w", "sensor": "rplidar_s2e", "min_range_m": -1}, "min_range_m"),
        ({"world": "w", "sensor": "rplidar_s2e", "max_range_m": 0}, "max_range_m"),
        ({"world": "w", "sensor": "rplidar_s2e", "min_range_m": 5, "max_range_m": 2}, "min_range_m"),
    ],
)
def test_validate_rejects(attrs, match):
    with pytest.raises(ValueError, match=match):
        IsaacLidar.validate_config(_config("lid", attrs))


def test_validate_ok():
    assert IsaacLidar.validate_config(
        _config("lid", {"world": "sim-world", "sensor": "ouster_os1_32", "max_range_m": 20})
    ) == (["sim-world"], [])


def test_lidar_frame_is_local_when_parented():
    cfg = _config("lid", {"world": "w", "sensor": "rplidar_s2e", "parent_prim": "/World/bot"})
    cfg.frame.parent = "my-base"
    cfg.frame.translation.z = 200
    attrs = apply_lidar_frame_to_attrs(cfg, get_attrs(cfg))
    assert attrs["local_position"] == pytest.approx([0, 0, 0.2])
    assert "position" not in attrs and "camera_axes" not in attrs


def test_point_cloud_filtered_and_errors(manager):
    lid = IsaacLidar.new(
        _config("lid", {"world": "sim-world", "sensor": "rplidar_s2e", "min_range_m": 0.5, "max_range_m": 5}),
        {},
    )
    with pytest.raises(RuntimeError, match="no lidar scan yet"):
        asyncio.run(lid.get_point_cloud())
    manager._mock_sim_time_s = 1.0
    data, mime = asyncio.run(lid.get_point_cloud())
    assert mime == "pointcloud/pcd"
    header, arr = parse_pcd(data)
    assert header["FIELDS"] == "x y z"
    assert len(arr) == 180  # only the 1 m returns
    assert "min_range_m" not in manager._handles["lid"][0]
    with pytest.raises(NotImplementedError, match="no image"):
        asyncio.run(lid.get_image())
    with pytest.raises(NotImplementedError, match="no image"):
        asyncio.run(lid.get_images())
    assert asyncio.run(lid.get_properties()).supports_pcd is True
    assert asyncio.run(lid.get_geometries()) == []


def test_validate_rejects_world_parent_with_parent_prim():
    cfg = _config("lid", {"world": "w", "sensor": "rplidar_s2e", "parent_prim": "/World/bot"})
    cfg.frame.parent = "world"
    with pytest.raises(ValueError, match="parent_prim"):
        IsaacLidar.validate_config(cfg)


def test_failed_reconfigure_keeps_state(manager):
    lid = IsaacLidar.new(
        _config("lid", {"world": "sim-world", "sensor": "rplidar_s2e", "max_range_m": 5}), {}
    )
    with pytest.raises(ValueError):
        lid.reconfigure(
            _config("lid", {"world": "sim-world", "sensor": "ouster_os1_32", "max_range_m": 2}), {}
        )
    assert lid._max_range_m == 5.0


class _FakeAnnotator:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return {"data": self._data}


def test_read_lidar_scan_drops_no_return_origin_points():
    pts = np.array([[0, 0, 0], [1, 2, 0], [0, 0, 0]], dtype=np.float32)
    out = _read_lidar_scan(_FakeAnnotator(pts))
    assert out.tolist() == [[1, 2, 0]]


def test_read_lidar_scan_all_zero_is_empty_not_none():
    # every beam missed is a real scan, distinct from "no data yet"
    out = _read_lidar_scan(_FakeAnnotator(np.zeros((3, 3), dtype=np.float32)))
    assert out.shape == (0, 3)
    assert _read_lidar_scan(_FakeAnnotator(None)) is None


def _isaac_handle(data):
    h = IsaacLidarHandle.__new__(IsaacLidarHandle)  # skip the Isaac-only __init__
    h._annotator = _FakeAnnotator(data)
    h._rate = h._native_hz = 10.0
    h._scan = None
    return h


def test_poll_empty_scan_replaces_old_one():
    h = _isaac_handle(np.array([[1, 0, 0]], dtype=np.float32))
    h.poll(0.1)
    assert len(h.latest_scan()[1]) == 1
    h._annotator = _FakeAnnotator(np.zeros((4, 3), dtype=np.float32))
    h.poll(0.2)
    assert h.latest_scan()[0] == 0.2 and len(h.latest_scan()[1]) == 0


def test_poll_after_reset_waits_for_a_full_rotation():
    h = _isaac_handle(np.array([[1, 0, 0]], dtype=np.float32))
    h.poll(5.0)
    h.poll(0.02)  # world reset: sim time went backwards
    assert h.latest_scan() is None
    h.poll(0.1)
    assert h.latest_scan()[0] == 0.1

