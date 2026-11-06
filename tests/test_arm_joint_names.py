"""`joint_names` validation and several arms sharing one Isaac articulation,
driven against stand-ins for the isaac namespace."""

from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
import numpy as np

from isaac_module import sim_manager
from isaac_module.sim_manager import SimManager, _merge_position_gains

HUMANOID_DOFS = (
    [f"torso_j{i}" for i in range(1, 4)]
    + [f"head_j{i}" for i in range(1, 4)]
    + [f"L_arm_j{i}" for i in range(1, 8)]
    + [f"R_arm_j{i}" for i in range(1, 8)]
)


def _config(attrs: dict) -> ComponentConfig:
    return ComponentConfig(
        name="a",
        attributes=dict_to_struct({"world": "w", "prim_path": "/W/r", **attrs}),
    )


def test_validate_accepts_joint_names():
    IsaacArm.validate_config(_config({"joint_names": ["a", "b"]}))


@pytest.mark.parametrize(
    "value, match",
    (
        ([], "non-empty list"),
        ("torso_j1", "non-empty list"),
        (["a", 1], "non-empty list"),
        (["a", ""], "non-empty list"),
        (["a", "a"], "unique"),
    ),
)
def test_validate_rejects_bad_joint_names(value, match):
    with pytest.raises(ValueError, match=match):
        IsaacArm.validate_config(_config({"joint_names": value}))


class _Art:
    def __init__(self, prim_path, name):
        self.prim_path = prim_path
        self.name = name
        self.dof_names = list(HUMANOID_DOFS)

    def set_default_state(self, **pose):
        pass

    def set_world_pose(self, **pose):
        pass


def _manager():
    manager = object.__new__(SimManager)
    calls = []
    manager._arm_articulations = {}
    manager._arm_usd_owners = {}
    manager._handles = {}
    manager._shared_bases = []
    manager.world = SimpleNamespace(
        scene=SimpleNamespace(add=lambda art: calls.append(("add", art.name))),
        reset=lambda: calls.append("reset"),
    )
    manager._isaac = SimpleNamespace(
        add_reference_to_stage=lambda **kw: calls.append(("usd", kw["prim_path"])),
        get_prim_at_path=lambda path: SimpleNamespace(IsValid=lambda: True),
        SingleXFormPrim=lambda path: SimpleNamespace(
            set_world_pose=lambda **pose: calls.append(("pose", path, pose))
        ),
        SingleArticulation=_Art,
    )
    return manager, calls


def test_second_arm_on_prim_reuses_articulation_without_reset_or_repose():
    manager, calls = _manager()
    torso = manager._create_arm_isaac(
        "torso",
        {
            "usd_path": "/robots/vega.usd",
            "prim_path": "/World/vega",
            "position": [1.0, 2.0, 0.0],
            "joint_names": ["torso_j1", "torso_j2", "torso_j3"],
        },
    )
    spawn_calls = list(calls)
    left = manager._create_arm_isaac(
        "left_arm",
        {
            "prim_path": "/World/vega",
            "joint_names": [f"L_arm_j{i}" for i in range(1, 8)],
        },
    )

    assert "reset" in spawn_calls
    assert calls == spawn_calls  # no re-pose to the origin, no scene.add, no reset
    assert left._art is torso._art
    assert torso._joint_indices == [0, 1, 2]
    assert left._joint_indices == list(range(6, 13))


def test_joint_names_default_to_every_dof():
    manager, _ = _manager()
    arm = manager._create_arm_isaac("whole", {"prim_path": "/World/vega"})
    assert arm.joint_names() == tuple(HUMANOID_DOFS)


def test_unknown_joint_error_names_the_arm():
    manager, _ = _manager()
    with pytest.raises(ValueError, match=r"arm 'head'.*\['neck'\] not found"):
        manager._create_arm_isaac(
            "head", {"prim_path": "/World/vega", "joint_names": ["neck"]}
        )


def test_only_one_arm_per_prim_may_load_the_usd():
    manager, _ = _manager()
    attrs = {"usd_path": "/robots/vega.usd", "prim_path": "/World/vega"}
    manager._create_arm_isaac("torso", attrs)
    with pytest.raises(ValueError, match="only one arm per prim_path"):
        manager._create_arm_isaac("head", attrs)


def test_closed_loader_arm_releases_prim_for_its_replacement():
    manager, calls = _manager()
    attrs = {"usd_path": "/robots/vega.usd", "prim_path": "/World/vega"}
    old = manager._create_arm_isaac("robot", attrs)
    spawn_calls = list(calls)

    manager.remove_arm("robot")  # what IsaacArm.close does on rename/removal
    new = manager._create_arm_isaac("torso", attrs)

    assert new._art is old._art
    assert calls == spawn_calls  # prims already in the stage: no reload, no reset
    with pytest.raises(ValueError, match="only one arm per prim_path"):
        manager._create_arm_isaac("head", attrs)  # the new loader holds the claim


WHEELS = ("L_wheel_j2", "R_wheel_j2")
VEGA_DOFS = list(WHEELS) + list(HUMANOID_DOFS)


def test_merge_position_gains_keeps_shape_and_wheels():
    kps, kds = np.zeros((1, 4)), np.full((1, 4), 300.0)
    out_kp, out_kd = _merge_position_gains(
        ["L_wheel_j2", "R_wheel_j2", "torso_j1", "head_j1"],
        kps,
        kds,
        {"torso_j1": (1e4, 300.0), "head_j1": (50.0, 5.0), "not_a_dof": (1.0, 1.0)},
    )
    assert out_kp.shape == (1, 4)
    assert out_kp.tolist() == [[0.0, 0.0, 1e4, 50.0]]
    assert out_kd.tolist() == [[300.0, 300.0, 300.0, 5.0]]
    assert kps.tolist() == [[0.0] * 4]  # inputs untouched


class _Controller:
    def __init__(self, n):
        self.kps, self.kds = np.full((1, n), 7.0), np.full((1, n), 1.0)

    def get_gains(self):
        return self.kps, self.kds

    def set_gains(self, kps, kds):
        self.kps, self.kds = np.array(kps), np.array(kds)

    def velocity_mode(self):  # what WheeledRobot.initialize does to the whole articulation
        self.kps = np.zeros_like(self.kps)


def test_base_sharing_articulation_keeps_arm_position_gains(monkeypatch):
    manager, calls = _manager()
    controller = _Controller(len(VEGA_DOFS))
    robot = SimpleNamespace(
        dof_names=VEGA_DOFS, name="base", get_articulation_controller=lambda: controller
    )
    manager.world.reset = lambda: (calls.append("reset"), controller.velocity_mode())
    manager.world.add_physics_callback = lambda *a: None
    manager._isaac.WheeledRobot = lambda **kw: robot
    manager._isaac.DifferentialController = lambda **kw: SimpleNamespace()
    usd = {name: (1e4, 300.0) for name in HUMANOID_DOFS}
    monkeypatch.setattr(
        sim_manager, "_usd_drive_gains", lambda prim, names: {n: usd[n] for n in names}
    )
    art = _Art("/World/vega", "torso")
    art.dof_names = VEGA_DOFS
    manager._isaac.SingleArticulation = lambda prim_path, name: art

    manager._create_base_isaac("base", {"prim_path": "/World/vega", "wheel_joints": list(WHEELS)})
    manager._create_arm_isaac(
        "torso", {"prim_path": "/World/vega", "joint_names": ["torso_j1", "torso_j2", "torso_j3"]}
    )

    assert calls.count("reset") == 2  # base init and first arm attach both re-zero gains
    kps = controller.kps.reshape(-1)
    assert kps[:2].tolist() == [0.0, 0.0]  # wheels stay velocity-controlled
    assert kps[2:].tolist() == [1e4] * len(HUMANOID_DOFS)
