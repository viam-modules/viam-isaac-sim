import asyncio
import json
import urllib.error

import pytest
from viam.components.arm import KinematicsFileFormat
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm

# a mock sim is all these tests need
from isaac_module.sim_manager import SimConfig, SimManager


@pytest.fixture(scope="module")
def sim():
    manager = SimManager.get()
    manager.configure(SimConfig(mock=True))
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
