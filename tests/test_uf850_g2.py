"""The UFACTORY 850 + G2 ships inside the module, not on Isaac's content server."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import sim_manager
from isaac_module.models.arm import IsaacArm
from isaac_module.sim_manager import MODULE_ASSETS_DIR, SimManager, gripper_meta

SHIPPED_USD = MODULE_ASSETS_DIR / "uf850_g2" / "uf850_g2.usd"


def _resolver() -> SimManager:
    manager = object.__new__(SimManager)
    manager._isaac = SimpleNamespace(
        get_assets_root_path=lambda: pytest.fail("content server was consulted")
    )
    manager._usd_exists = lambda path: Path(path).is_file()
    return manager


def test_uf850_g2_resolves_to_the_usd_shipped_with_the_module():
    usd, meta = _resolver()._resolve_usd({"asset": "uf850_g2"})
    assert Path(usd) == SHIPPED_USD
    assert SHIPPED_USD.is_file()
    assert meta["end_effector_link"] == "link6"


def test_usd_path_still_overrides_the_shipped_usd():
    usd, _ = _resolver()._resolve_usd({"asset": "uf850_g2", "usd_path": __file__})
    assert usd == __file__


def test_missing_shipped_usd_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(sim_manager, "MODULE_ASSETS_DIR", tmp_path)
    with pytest.raises(ValueError, match="usd not found"):
        _resolver()._resolve_usd({"asset": "uf850_g2"})


def _pad_gap_mm(drive_rad: float) -> float:
    # UFACTORY's G2 linkage (xarm_ros2 xarm_gripper.urdf.xacro) with parallel pads.
    phase = math.atan2(35.465, 42.039)
    return 18.0 + 110.0 * math.sin(phase - drive_rad)


def test_g2_drive_range_spans_the_measured_real_pad_gap():
    meta = gripper_meta({"asset": "g2"})
    assert meta["joints_on_arm"]
    assert meta["drive_joint"] == "drive_joint"
    assert _pad_gap_mm(meta["open_position"]) == pytest.approx(95.0, abs=0.01)
    assert _pad_gap_mm(meta["closed_position"]) == pytest.approx(0.0, abs=0.01)


@pytest.fixture
def manager(monkeypatch):
    value = SimManager()
    value.mock = True
    value._booted.set()
    monkeypatch.setattr(SimManager, "_instance", value)
    return value


def _arm(asset: str) -> None:
    attrs = {"world": "sim-world", "asset": asset, "mock_dof": 6}
    IsaacArm.new(ComponentConfig(name="pick-arm", attributes=dict_to_struct(attrs)), {})


def test_g2_requires_the_uf850_g2_arm(manager):
    _arm("ur20")
    with pytest.raises(ValueError, match="requires arm asset 'uf850_g2'"):
        manager.create_gripper("g", {"arm": "pick-arm", "asset": "g2"})


def test_g2_attaches_to_the_uf850_g2_arm(manager):
    _arm("uf850_g2")
    assert manager.create_gripper("g", {"arm": "pick-arm", "asset": "g2"}) is not None
