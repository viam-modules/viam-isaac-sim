"""Arms can spawn at a configured joint pose instead of zero joints, which
can interpenetrate a neighboring arm or the floor."""

import math
import sys
from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import sim_manager
from isaac_module.models.arm import IsaacArm
from isaac_module.sim_manager import (
    KNOWN_ASSETS,
    UR_JOINT_NAMES,
    MockArmHandle,
    SimManager,
    _author_initial_joint_state,
)

HOME_DEG = [102.1, -158.65, -98.84, 77.35, 78.32, 90.47]


class _Joint:
    pass


class _RevoluteJoint(_Joint):
    pass


class _PrismaticJoint(_Joint):
    pass


class _Prim:
    def __init__(self, path, kind=None, children=()):
        self.path = path
        self.kind = kind
        self.children = list(children)

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def GetPath(self):
        return self.path

    def IsA(self, cls):
        return self.kind is not None and issubclass(self.kind, cls)


@pytest.fixture
def authored(monkeypatch):
    """Fake pxr recording what each joint's drive and joint state receive."""
    values = {}

    def _recorder(prim, prefix):
        def create(attr):
            return lambda value: values.__setitem__(
                (prim.GetName(), f"{prefix}.{attr}"), value
            )

        return SimpleNamespace(
            CreateTargetPositionAttr=create("target"),
            CreatePositionAttr=create("position"),
            CreateVelocityAttr=create("velocity"),
        )

    def prim_range(root):
        stack = [root]
        while stack:
            prim = stack.pop()
            yield prim
            stack.extend(prim.children)

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Usd=SimpleNamespace(PrimRange=prim_range),
            UsdPhysics=SimpleNamespace(
                Joint=_Joint,
                RevoluteJoint=_RevoluteJoint,
                DriveAPI=SimpleNamespace(
                    Apply=lambda prim, axis: _recorder(prim, f"drive:{axis}")
                ),
            ),
            PhysxSchema=SimpleNamespace(
                JointStateAPI=SimpleNamespace(
                    Apply=lambda prim, axis: _recorder(prim, f"state:{axis}")
                )
            ),
        ),
    )
    return values


def _ur_prim(kinds=None):
    kinds = kinds or {}
    joints = [
        _Prim(f"/World/arm/joints/{name}", kinds.get(name, _RevoluteJoint))
        for name in UR_JOINT_NAMES
    ]
    links = [_Prim("/World/arm/base_link"), _Prim("/World/arm/wrist_3_link")]
    return _Prim(
        "/World/arm", children=[*links, _Prim("/World/arm/joints", children=joints)]
    )


def test_initial_pose_is_authored_as_joint_state_and_drive_target(authored):
    _author_initial_joint_state(_ur_prim(), UR_JOINT_NAMES, HOME_DEG)

    for name, degrees in zip(UR_JOINT_NAMES, HOME_DEG):
        assert authored[(name, "drive:angular.target")] == degrees
        assert authored[(name, "state:angular.position")] == degrees
        assert authored[(name, "state:angular.velocity")] == 0.0


def test_initial_pose_must_cover_every_arm_joint(authored):
    with pytest.raises(ValueError, match="needs 6 values"):
        _author_initial_joint_state(_ur_prim(), UR_JOINT_NAMES, HOME_DEG[:5])
    assert authored == {}


def test_initial_pose_rejects_a_missing_joint(authored):
    root = _ur_prim()
    root.children[-1].children.pop()  # drop wrist_3_joint
    with pytest.raises(ValueError, match="wrist_3_joint"):
        _author_initial_joint_state(root, UR_JOINT_NAMES, HOME_DEG)


def test_initial_pose_rejects_a_non_revolute_joint(authored):
    root = _ur_prim({"elbow_joint": _PrismaticJoint})
    with pytest.raises(ValueError, match="elbow_joint.*not revolute"):
        _author_initial_joint_state(root, UR_JOINT_NAMES, HOME_DEG)


def _spawn(monkeypatch, attrs):
    """Run _create_arm_isaac against fakes; return what was authored and the
    kwargs the arm handle was built with."""
    calls = {}
    manager = object.__new__(SimManager)
    manager._arm_articulations = {}
    manager._arm_usd_owners = {}
    manager._shared_bases = []
    manager._resolve_usd = lambda a: ("/fake/arm.usd", KNOWN_ASSETS.get(a.get("asset"), {}))
    manager._require_prim = lambda path: None
    manager._isaac = SimpleNamespace(
        add_reference_to_stage=lambda usd_path, prim_path: None,
        get_prim_at_path=lambda path: path,
        SingleXFormPrim=lambda path: SimpleNamespace(set_world_pose=lambda **pose: None),
        SingleArticulation=lambda prim_path, name: SimpleNamespace(),
    )
    manager.world = SimpleNamespace(scene=SimpleNamespace(add=lambda art: None))
    monkeypatch.setattr(
        sim_manager, "_reset_articulation_at_pose", lambda *args: None
    )
    monkeypatch.setattr(
        sim_manager,
        "_author_initial_joint_state",
        lambda prim, names, degrees: calls.update(authored=(prim, names, degrees)),
    )
    monkeypatch.setattr(
        sim_manager,
        "IsaacArmHandle",
        lambda *args, **kwargs: calls.update(handle_kwargs=kwargs),
    )
    manager._create_arm_isaac("arm", attrs)
    return calls


def test_spawn_authors_the_initial_pose_and_seeds_the_commanded_target(monkeypatch):
    calls = _spawn(
        monkeypatch, {"asset": "ur20", "initial_joint_positions_deg": HOME_DEG}
    )
    assert calls["authored"] == ("/World/arm", UR_JOINT_NAMES, HOME_DEG)
    assert calls["handle_kwargs"]["initial_target"] == pytest.approx(
        [math.radians(v) for v in HOME_DEG]
    )


def test_spawn_without_an_initial_pose_is_unchanged(monkeypatch):
    calls = _spawn(monkeypatch, {"asset": "ur20"})
    assert "authored" not in calls
    assert calls["handle_kwargs"]["initial_target"] is None


def test_initial_pose_needs_known_joint_names(monkeypatch):
    with pytest.raises(ValueError, match="joint_names or a known asset"):
        _spawn(monkeypatch, {"initial_joint_positions_deg": HOME_DEG})


def test_mock_arm_starts_at_the_initial_pose():
    handle = MockArmHandle(
        None, "arm", {"asset": "ur20", "initial_joint_positions_deg": HOME_DEG}
    )
    assert handle.get_joint_positions() == pytest.approx(
        [math.radians(v) for v in HOME_DEG]
    )
    assert not handle.is_moving()


def test_mock_arm_rejects_an_initial_pose_of_the_wrong_length():
    with pytest.raises(ValueError, match="needs 6 values"):
        MockArmHandle(None, "arm", {"initial_joint_positions_deg": [0.0, 1.0]})


@pytest.mark.parametrize(
    "value", ("0", [], [0, "1"], [True, 0], [0, float("inf")], [float("nan")])
)
def test_initial_pose_must_be_a_list_of_finite_numbers(value):
    config = ComponentConfig(
        name="arm",
        attributes=dict_to_struct(
            {
                "world": "sim-world",
                "asset": "ur20",
                "initial_joint_positions_deg": value,
            }
        ),
    )

    with pytest.raises(ValueError, match="initial_joint_positions_deg"):
        IsaacArm.validate_config(config)
