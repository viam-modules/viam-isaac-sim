import asyncio
import threading

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.scene_camera import META_PATH, IsaacSceneCamera
from isaac_module.sim_manager import SimManager


def _config(name, attrs):
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture
def manager():
    """Mock SimManager installed as the singleton; sim-thread work runs inline."""
    previous = SimManager._instance
    m = SimManager()
    m.mock = True
    m._booted.set()
    m._sim_thread_id = threading.get_ident()
    SimManager._instance = m
    yield m
    SimManager._instance = previous


def _camera(attrs, name="overview"):
    return IsaacSceneCamera.new(_config(name, {"world": "sim-world", **attrs}), {})


# ---- validation ----

def test_validate_defaults_pass():
    assert IsaacSceneCamera.validate_config(_config("c", {"world": "sim-world"})) == (["sim-world"], [])


def test_validate_rejects_parent_prim():
    with pytest.raises(ValueError, match="parent_prim"):
        IsaacSceneCamera.validate_config(
            _config("c", {"world": "sim-world", "parent_prim": "/World/arm/wrist"})
        )


@pytest.mark.parametrize("key", ["orientation_wxyz", "orientation_rpy_deg"])
def test_validate_rejects_orientation_attrs(key):
    value = [1, 0, 0, 0] if key == "orientation_wxyz" else [0, 0, 0]
    with pytest.raises(ValueError, match=f"attributes.{key}"):
        IsaacSceneCamera.validate_config(_config("c", {"world": "sim-world", key: value}))


def test_validate_rejects_coincident_position_and_target():
    with pytest.raises(ValueError, match="must differ"):
        IsaacSceneCamera.validate_config(
            _config("c", {"world": "sim-world", "position": [1, 1, 1], "target": [1, 1, 1]})
        )


def test_validate_rejects_straight_down_home():
    with pytest.raises(ValueError, match="89"):
        IsaacSceneCamera.validate_config(
            _config("c", {"world": "sim-world", "position": [0, 0, 3], "target": [0, 0, 0]})
        )


def test_validate_accepts_translation_only_and_identity_frames():
    config = _config("c", {"world": "sim-world"})
    config.frame.parent = "world"
    config.frame.translation.x = 2000
    config.frame.translation.z = 1000
    IsaacSceneCamera.validate_config(config)

    config.frame.orientation.quaternion.w = 1
    IsaacSceneCamera.validate_config(config)


def test_validate_rejects_rotated_frame():
    config = _config("c", {"world": "sim-world"})
    config.frame.parent = "world"
    config.frame.translation.x = 2000
    config.frame.orientation.quaternion.w = 0.70710678
    config.frame.orientation.quaternion.z = 0.70710678
    with pytest.raises(ValueError, match="frame.orientation"):
        IsaacSceneCamera.validate_config(config)


def test_validate_rejects_bool_fov():
    with pytest.raises(ValueError, match="fov_deg"):
        IsaacSceneCamera.validate_config(_config("c", {"world": "sim-world", "fov_deg": True}))


def test_validate_rejects_position_with_frame():
    config = _config("c", {"world": "sim-world", "position": [1, 2, 3]})
    config.frame.parent = "world"
    with pytest.raises(ValueError, match="attributes.position conflicts"):
        IsaacSceneCamera.validate_config(config)


# ---- home view ----

def test_home_view_from_config(manager):
    cam = _camera({"position": [3, 0, 0], "target": [0, 0, 0]})
    view = cam.get_view()
    assert view["distance"] == pytest.approx(3.0)
    assert view["yaw_deg"] == pytest.approx(0.0)
    assert view["pitch_deg"] == pytest.approx(0.0)
    assert view["fov_deg"] == pytest.approx(70.0)
    assert view["target"] == [0.0, 0.0, 0.0]
    assert view["home"] == {k: view[k] for k in ("target", "distance", "yaw_deg", "pitch_deg", "fov_deg")}
    assert view["recording"] is False


def test_reconfigure_always_writes_pose_keys(manager):
    _camera({})
    attrs = manager._handles["overview"][0]
    assert attrs["position"] == [3.0, 3.0, 2.5]
    assert attrs["target"] == [0.0, 0.0, 0.0]
    assert attrs["fov_deg"] == 70.0
    assert "orientation_wxyz" not in attrs


def test_frame_translation_is_home_position(manager):
    config = _config("overview", {"world": "sim-world"})
    config.frame.parent = "world"
    config.frame.translation.x = 2000
    config.frame.orientation.quaternion.w = 1
    cam = IsaacSceneCamera.new(config, {})
    assert cam.get_view()["distance"] == pytest.approx(2.0)
    assert "orientation_wxyz" not in manager._handles["overview"][0]


def test_meta_path_points_at_repo_meta_json():
    assert META_PATH.name == "meta.json"
    assert META_PATH.is_file()


# ---- set_view / reset_view ----

def test_set_view_merges_and_moves_camera(manager):
    cam = _camera({"position": [3, 0, 0], "target": [0, 0, 0]})
    view = cam.set_view({"yaw_deg": 90.0})
    assert view["yaw_deg"] == pytest.approx(90.0)
    assert view["distance"] == pytest.approx(3.0)  # untouched keys kept
    attrs = manager._handles["overview"][0]
    assert attrs["position"] == pytest.approx([0.0, 3.0, 0.0], abs=1e-9)
    assert attrs["target"] == [0.0, 0.0, 0.0]


def test_set_view_clamps(manager):
    cam = _camera({"position": [3, 0, 0]})
    view = cam.set_view({"pitch_deg": 120.0, "distance": 0.0, "yaw_deg": 270.0, "fov_deg": 500.0})
    assert view["pitch_deg"] == 89.0
    assert view["distance"] == 0.05
    assert view["yaw_deg"] == pytest.approx(-90.0)
    assert view["fov_deg"] == 179.0


def test_set_view_target_and_fov(manager):
    cam = _camera({"position": [3, 0, 0]})
    view = cam.set_view({"target": [1, 2, 3], "fov_deg": 40})
    assert view["target"] == [1.0, 2.0, 3.0]
    attrs = manager._handles["overview"][0]
    assert attrs["fov_deg"] == 40.0
    assert attrs["position"] == pytest.approx([4.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "patch, match",
    [
        ({"zoom": 2}, "unknown keys"),
        ({"distance": "far"}, "distance must be a finite number"),
        ({"yaw_deg": True}, "yaw_deg must be a finite number"),
        ({"target": [1, 2]}, "target must be"),
        ({"target": None}, "target must be"),
        ({"target": "123"}, "target must be"),
        ({"target": [0, 0, float("nan")]}, "target must be"),
    ],
)
def test_set_view_rejects_bad_input(manager, patch, match):
    cam = _camera({})
    before = cam.get_view()
    with pytest.raises(ValueError, match=match):
        cam.set_view(patch)
    assert cam.get_view() == before


def test_reset_view_returns_home(manager):
    cam = _camera({"position": [3, 0, 0]})
    cam.set_view({"yaw_deg": 45.0, "distance": 1.0})
    view = cam.reset_view()
    assert view["yaw_deg"] == pytest.approx(0.0)
    assert view["distance"] == pytest.approx(3.0)
    assert manager._handles["overview"][0]["position"] == pytest.approx([3.0, 0.0, 0.0])


def test_recording_blocks_and_reports(manager):
    cam = _camera({"position": [3, 0, 0]})
    # fly first: a reset_view while already at home is a no-op that create_camera never rejects
    cam.set_view({"yaw_deg": 30.0})
    before = cam.get_view()
    manager._episode_id = "000001"
    with pytest.raises(RuntimeError, match="during episode capture"):
        cam.set_view({"yaw_deg": 45.0})
    with pytest.raises(RuntimeError, match="during episode capture"):
        cam.reset_view()
    after = cam.get_view()
    assert after["recording"] is True
    assert {k: after[k] for k in before if k != "recording"} == {
        k: before[k] for k in before if k != "recording"
    }


def test_reconfigure_after_flying_snaps_home_without_restart(manager):
    # config without target/fov_deg: regression for create_camera's
    # "removed key requires a module restart" rules
    config = _config("overview", {"world": "sim-world", "position": [3, 0, 1]})
    cam = IsaacSceneCamera.new(config, {})
    cam.set_view({"yaw_deg": 90.0, "fov_deg": 50.0})
    cam.reconfigure(config, {})  # must not raise
    view = cam.get_view()
    assert view["yaw_deg"] == pytest.approx(0.0)
    assert view["fov_deg"] == 70.0
    assert manager._handles["overview"][0]["position"] == [3.0, 0.0, 1.0]


# ---- do_command ----

def test_do_command_unknown(manager):
    cam = _camera({})
    with pytest.raises(ValueError, match="unknown command 'nope'"):
        asyncio.run(cam.do_command({"command": "nope"}))


def test_do_command_serves_custom_control(manager):
    if not (META_PATH.parent / "scene_camera_ui" / "dist" / "index.html").is_file():
        pytest.skip("scene_camera_ui not built (the UI CI job covers the artifact)")
    cam = _camera({})
    served = asyncio.run(cam.do_command({"command": "get-custom-control"}))
    assert served["path"] == "scene_camera_ui/dist/index.html"
    assert served["html"].lstrip().lower().startswith("<!doctype html")
    assert len(served["revision"]) == 64
