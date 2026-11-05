"""`joint_names` validation and several arms sharing one Isaac articulation,
driven against stand-ins for the isaac namespace."""

from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.sim_manager import SimManager

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
