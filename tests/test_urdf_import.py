"""A `.urdf` usd_path is imported with Isaac's URDF importer into a cached
USD, and the arm loads that USD."""

import os
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

    def fake_import(urdf_path, dest_path, fix_base, drive):
        calls.append((urdf_path, dest_path, fix_base, drive))
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


def test_urdf_usd_path_is_imported_and_the_arm_loads_the_usd(imports, tmp_path):
    robot = _urdf(tmp_path)
    manager = _manager()
    manager._arm_articulations = {}
    referenced = []

    def add_reference_to_stage(usd_path, prim_path):
        referenced.append(usd_path)
        raise StopIteration  # the rest of the spawn isn't under test

    manager._isaac = SimpleNamespace(add_reference_to_stage=add_reference_to_stage)
    with pytest.raises(StopIteration):
        manager._create_arm_isaac("arm", {"usd_path": str(robot)})

    [(urdf_path, dest_path, _, _)] = imports
    # absolute, so the importer resolves meshes against the URDF's directory
    assert urdf_path == str(robot.resolve())
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

    def failing(urdf_path, dest_path, fix_base, drive):
        attempts.append(dest_path)
        # a retry starts clean: the importer would reopen a half-written stage
        assert not os.path.exists(dest_path)
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


def test_urdf_drive_is_passed_to_the_importer_and_keys_the_cache(imports, tmp_path):
    robot = _urdf(tmp_path)
    manager = _manager()
    default, _ = manager._resolve_usd({"usd_path": str(robot)})
    soft, _ = manager._resolve_usd(
        {"usd_path": str(robot), "urdf_drive": {"natural_frequency": 25, "damping_ratio": 0.5}}
    )
    assert soft != default
    assert [call[3] for call in imports] == [(750.0, 1.0), (25.0, 0.5)]


@pytest.mark.parametrize(
    "drive",
    [
        {"natural_frequency": 0},
        {"natural_frequency": float("nan")},
        {"damping_ratio": -1},
        {"stiffness": 5},
        "fast",
    ],
)
def test_bad_urdf_drive_is_rejected(imports, tmp_path, drive):
    robot = _urdf(tmp_path)
    with pytest.raises(ValueError, match="urdf_drive"):
        _manager()._resolve_usd({"usd_path": str(robot), "urdf_drive": drive})
    assert imports == []


def test_drive_gains_follow_the_importers_natural_frequency_math():
    """Same as the importer UI: stiffness from compute_natural_stiffness,
    damping = 2 * m_eq * natural_frequency * damping_ratio, where m_eq is 1
    for acceleration drives and the joint inertia for force drives. The
    importer writes gains as USD values, which are per degree for revolute
    joints, so those are converted from per radian; prismatic stay SI."""
    import math

    position, force_type = "position", "force"
    revolute, continuous, prismatic = "revolute", "continuous", "prismatic"
    urdf = SimpleNamespace(
        UrdfJointTargetType=SimpleNamespace(JOINT_DRIVE_POSITION=position),
        UrdfJointDriveType=SimpleNamespace(JOINT_DRIVE_FORCE=force_type),
        UrdfJointType=SimpleNamespace(
            JOINT_REVOLUTE=revolute,
            JOINT_CONTINUOUS=continuous,
            JOINT_PRISMATIC=prismatic,
        ),
    )

    def joint(target_type, drive_type, inertia, kind=revolute):
        drive = SimpleNamespace(target_type=target_type, drive_type=drive_type)
        return SimpleNamespace(drive=drive, inertia=inertia, type=kind)

    joints = {
        "torso": joint(position, "acceleration", 7.0),
        "wheel": joint(position, force_type, 0.5, continuous),
        "lift": joint(position, "acceleration", 3.0, prismatic),
        "free": joint("none", "acceleration", 1.0),
    }
    robot = SimpleNamespace()
    robot.joints = joints
    interface = SimpleNamespace(
        compute_natural_stiffness=lambda r, name, nf: {"torso": 1, "wheel": 2, "lift": 1}[name]
        * nf
        * nf
    )
    sim_manager._set_urdf_drives(urdf, interface, robot, 100.0, 1.0)

    torso, wheel, lift, free = (
        robot.joints[n].drive for n in ("torso", "wheel", "lift", "free")
    )
    per_degree = math.pi / 180
    assert (torso.strength, torso.damping) == pytest.approx(
        (10000.0 * per_degree, 200.0 * per_degree)
    )
    assert (wheel.strength, wheel.damping) == pytest.approx(
        (20000.0 * per_degree, 100.0 * per_degree)
    )
    assert (lift.strength, lift.damping) == (10000.0, 200.0)
    assert (torso.natural_frequency, torso.damping_ratio) == (100.0, 1.0)
    assert not hasattr(free, "strength")


def test_cold_urdf_import_runs_outside_the_spawn_timeout(monkeypatch):
    manager = object.__new__(SimManager)
    manager.mock = False
    manager._handles = {}
    manager._require_booted = lambda: None
    budgets = []

    def run(fn, timeout=30.0):
        budgets.append((fn.__name__ if hasattr(fn, "__name__") else fn, timeout))
        return fn()

    manager.run = run
    manager._resolve_usd = lambda attrs: ("/cache/bot.usd", {})
    manager._create_arm_isaac = lambda name, attrs: "arm-handle"
    manager._create_base_isaac = lambda name, attrs: "base-handle"

    assert manager.create_arm("arm", {"usd_path": "/r/bot.urdf"}) == "arm-handle"
    assert manager.create_base("base", {"usd_path": "/r/bot.URDF"}) == "base-handle"
    timeouts = [t for _, t in budgets]
    assert timeouts == [sim_manager.URDF_IMPORT_TIMEOUT_S, 120.0] * 2
    assert sim_manager.URDF_IMPORT_TIMEOUT_S >= 600

    budgets.clear()
    manager._handles = {}
    manager.create_arm("arm", {"usd_path": "/r/bot.usd"})
    assert [t for _, t in budgets] == [120.0]
