import asyncio
import json
import urllib.error

import pytest
from viam.components.arm import KinematicsFileFormat
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm

# reuse the booted mock sim from test_mock_sim's fixtures via our own tiny setup
from isaac_module.sim_manager import SimConfig, SimManager
import threading


@pytest.fixture(scope="module")
def sim():
    manager = SimManager.get()
    if not manager._booted.is_set():
        t = threading.Thread(target=manager.main_loop, daemon=True)
        t.start()
        manager.ensure_booted(SimConfig(mock=True))
    return manager


def _arm(name, attrs):
    return IsaacArm.new(
        ComponentConfig(name=name, attributes=dict_to_struct(attrs)), {}
    )


def test_kinematics_from_file_url(sim, tmp_path):
    sva = {"name": "test-arm", "kinematic_param_type": "SVA", "links": [], "joints": []}
    path = tmp_path / "test-arm.json"
    path.write_text(json.dumps(sva))

    arm = _arm(
        "kin-file-arm",
        {"world": "sim-world", "asset": "ur20", "kinematics_url": path.as_uri()},
    )
    fmt, data = asyncio.run(arm.get_kinematics())
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
    assert json.loads(data)["name"] == "test-arm"


def test_kinematics_urdf_format_detection(sim, tmp_path):
    path = tmp_path / "test-arm.urdf"
    path.write_text("<robot name='t'/>")
    arm = _arm(
        "kin-urdf-arm",
        {"world": "sim-world", "asset": "ur20", "kinematics_url": path.as_uri()},
    )
    fmt, _ = asyncio.run(arm.get_kinematics())
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF


def test_kinematics_missing_raises(sim):
    arm = _arm("kin-none-arm", {"world": "sim-world", "asset": "franka"})
    with pytest.raises(NotImplementedError, match="kinematics_url"):
        asyncio.run(arm.get_kinematics())


def test_kinematics_known_asset_download(sim):
    arm = _arm("kin-ur20-arm", {"world": "sim-world", "asset": "ur20"})
    try:
        fmt, data = asyncio.run(arm.get_kinematics())
    except urllib.error.URLError:
        pytest.skip("no network")
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
    parsed = json.loads(data)
    assert parsed["name"].lower() == "ur20"
    assert parsed["kinematic_param_type"] == "SVA"
    # cached: second call returns identical bytes
    fmt2, data2 = asyncio.run(arm.get_kinematics())
    assert data2 == data


def _derived_arm(name, monkeypatch, **attrs):
    from test_stage_geometry import _chain

    arm = _arm(name, {"world": "sim-world", "joint_names": ["j1"], **attrs})
    monkeypatch.setattr(arm._handle, "stage_chain", lambda name, visual: _chain(), raising=False)
    return arm


def test_arm_without_kinematics_url_derives_urdf_from_the_stage(sim, monkeypatch):
    arm = _derived_arm("kin-derived-arm", monkeypatch)
    fmt, data = asyncio.run(arm.get_kinematics())
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF
    assert b"<box" in data


def test_derived_arm_serves_mesh_collisions_with_its_kinematics(sim, monkeypatch):
    arm = _derived_arm("kin-mesh-arm", monkeypatch, planning_geometry="mesh")
    fmt, data, meshes = asyncio.run(arm.get_kinematics())
    assert b'filename="l1.ply"' in data
    assert meshes["l1.ply"].content_type == "ply"


def test_derived_arm_visual_models(sim, monkeypatch):
    pytest.importorskip("trimesh")
    arm = _derived_arm("kin-visual-arm", monkeypatch, visual_models=True)
    models = asyncio.run(arm.get_3d_models())
    assert list(models) == ["l1"]
    assert models["l1"].content_type == "model/gltf-binary"
    plain = _derived_arm("kin-novisual-arm", monkeypatch)
    assert asyncio.run(plain.get_3d_models()) == {}


@pytest.mark.parametrize(
    "attrs",
    [
        {"kinematics_url": "file:///x.json", "planning_geometry": "mesh"},
        {"kinematics_url": "file:///x.json", "visual_models": True},
        {"planning_geometry": "hull"},
        {"visual_models": "yes"},
    ],
)
def test_validate_rejects_stage_geometry_attrs_it_cannot_serve(attrs):
    config = ComponentConfig(
        name="bad", attributes=dict_to_struct({"world": "sim-world", **attrs})
    )
    with pytest.raises(ValueError):
        IsaacArm.validate_config(config)
