"""A `.urdf` usd_path is imported with Isaac's URDF importer into a cached
USD, and the arm loads that USD."""

import threading
from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import sim_manager
from isaac_module.models.arm import IsaacArm
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimManager

URDF = '<robot name="bot"><link name="base_link"/></robot>'


@pytest.fixture
def imports(monkeypatch, tmp_path):
    """Stub the Isaac importer: record each call and write the USD it would."""
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path / "data"))
    calls = []

    def fake_import(urdf_path, dest_path, fix_base):
        calls.append((urdf_path, dest_path, fix_base))
        with open(dest_path, "w") as f:
            f.write("#usda 1.0\n")

    monkeypatch.setattr(sim_manager, "_run_urdf_importer", fake_import)
    return calls


def _urdf(tmp_path, text=URDF):
    robot = tmp_path / "robot" / "bot.urdf"
    robot.parent.mkdir(exist_ok=True)
    robot.write_text(text)
    return robot


def _manager():
    manager = object.__new__(SimManager)
    manager._usd_exists = lambda path: True
    return manager


def test_urdf_usd_path_is_imported_and_the_arm_loads_the_usd(
    imports, tmp_path, monkeypatch
):
    robot = _urdf(tmp_path)
    manager = _manager()
    manager._arm_articulations = {}
    manager._arm_usd_owners = {}
    manager._shared_bases = []
    manager._require_prim = lambda path: None
    referenced = []
    manager._isaac = SimpleNamespace(
        add_reference_to_stage=lambda usd_path, prim_path: referenced.append(usd_path),
        get_prim_at_path=lambda path: path,
        SingleXFormPrim=lambda path: SimpleNamespace(set_world_pose=lambda **pose: None),
        SingleArticulation=lambda prim_path, name: SimpleNamespace(),
    )
    manager.world = SimpleNamespace(scene=SimpleNamespace(add=lambda art: None))
    manager._restore_shared_base_gains = lambda: None
    monkeypatch.setattr(sim_manager, "_reset_articulation_at_pose", lambda *a: None)
    monkeypatch.setattr(sim_manager, "IsaacArmHandle", lambda *a, **kw: None)
    manager._isaac_arm_handle = lambda *a: None

    manager._create_arm_isaac("arm", {"usd_path": str(robot), "urdf_fix_base": True})

    [(urdf_path, dest_path, fix_base)] = imports
    # absolute, so the importer resolves meshes against the URDF's directory
    assert urdf_path == str(robot.resolve())
    assert fix_base is True
    assert dest_path.startswith(str(tmp_path / "data"))
    assert dest_path.endswith(".usd")
    assert referenced == [dest_path]


def test_cached_import_is_reused_until_the_urdf_or_config_changes(imports, tmp_path):
    robot = _urdf(tmp_path)
    manager = _manager()
    first, _ = manager._resolve_usd({"usd_path": str(robot)})
    again, _ = manager._resolve_usd({"usd_path": str(robot)})
    assert again == first
    assert len(imports) == 1

    fixed, _ = manager._resolve_usd({"usd_path": str(robot), "urdf_fix_base": True})
    assert fixed != first
    robot.write_text(URDF.replace("base_link", "root"))
    edited, _ = manager._resolve_usd({"usd_path": str(robot)})
    assert edited not in (first, fixed)
    assert len(imports) == 3


def test_a_failed_import_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path / "data"))
    attempts = []

    def failing(urdf_path, dest_path, fix_base):
        attempts.append(dest_path)
        open(dest_path, "w").close()  # the importer creates the stage first
        raise RuntimeError("bad urdf")

    monkeypatch.setattr(sim_manager, "_run_urdf_importer", failing)
    robot = _urdf(tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="bad urdf"):
            _manager()._resolve_usd({"usd_path": str(robot)})
    assert len(attempts) == 2


def test_mock_mode_accepts_a_urdf_usd_path(imports, tmp_path):
    previous = SimManager._instance
    manager = SimManager()
    SimManager._instance = manager
    t = threading.Thread(target=manager.main_loop, daemon=True)
    t.start()
    try:
        cfg = lambda name, attrs: ComponentConfig(name=name, attributes=dict_to_struct(attrs))
        IsaacWorld.new(cfg("sim-world", {"mock": True}), {})
        attrs = {"world": "sim-world", "usd_path": "/robots/bot.urdf", "mock_dof": 1}
        IsaacArm.validate_config(cfg("torso", attrs))
        IsaacArm.new(cfg("torso", attrs), {})
    finally:
        manager.request_stop()
        t.join(timeout=5)
        SimManager._instance = previous
    assert imports == []
