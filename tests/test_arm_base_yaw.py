"""The spawned arm USD must share a base frame with the kinematics served by
GetKinematics, or the sim arm is mirrored from Viam's frame system."""

from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import sim_manager
from isaac_module.models.arm import IsaacArm
from isaac_module.sim_manager import KNOWN_ASSETS, SimManager, _arm_base_orientation
from isaac_module.spatial import quat_from_euler_deg, quat_rotate


def _rotate(orientation, v):
    return quat_rotate(tuple(orientation), v)


def test_only_ur_assets_with_controller_frame_kinematics_get_an_offset():
    # Measured against the pinned kinematics: Isaac 5.0 UR USDs reach +X at
    # zero joints; ur5e.json/ur20.json reach -X; ur3e.json reaches +X.
    offsets = {
        name: meta["base_yaw_offset_deg"]
        for name, meta in KNOWN_ASSETS.items()
        if "base_yaw_offset_deg" in meta
    }
    assert offsets == {"ur5e": 180.0, "ur20": 180.0}


def test_ur_kinematics_are_pinned_to_a_commit():
    for name in ("ur3e", "ur5e", "ur20"):
        assert "/main/" not in KNOWN_ASSETS[name]["kinematics"]


def test_zero_offset_leaves_orientation_untouched():
    assert _arm_base_orientation(None, 0.0) is None
    quat = quat_from_euler_deg(0.0, 0.0, 30.0)
    assert _arm_base_orientation(quat, 0.0) == pytest.approx(list(quat))


def test_offset_without_frame_orientation_turns_the_base_around():
    orientation = _arm_base_orientation(None, 180.0)
    assert _rotate(orientation, (1, 0, 0)) == pytest.approx((-1, 0, 0), abs=1e-9)
    assert _rotate(orientation, (0, 0, 1)) == pytest.approx((0, 0, 1), abs=1e-9)


def test_offset_is_applied_in_the_base_frame_after_the_frame_orientation():
    # A ceiling mount (roll 180) must keep the arm upside down; the offset
    # spins about the arm's own base Z, not world Z.
    ceiling = quat_from_euler_deg(180.0, 0.0, 0.0)
    orientation = _arm_base_orientation(ceiling, 180.0)
    assert _rotate(orientation, (1, 0, 0)) == pytest.approx((-1, 0, 0), abs=1e-9)
    assert _rotate(orientation, (0, 1, 0)) == pytest.approx((0, 1, 0), abs=1e-9)
    assert _rotate(orientation, (0, 0, 1)) == pytest.approx((0, 0, -1), abs=1e-9)

    yawed = _arm_base_orientation(quat_from_euler_deg(0.0, 0.0, 90.0), 180.0)
    assert _rotate(yawed, (1, 0, 0)) == pytest.approx((0, -1, 0), abs=1e-9)


def _spawn(monkeypatch, attrs):
    """Run _create_arm_isaac against fakes; return the USD base pose and the
    pose handed to the articulation reset."""
    poses = []
    resets = []
    manager = object.__new__(SimManager)
    manager._resolve_usd = lambda a: (
        "/fake/arm.usd",
        KNOWN_ASSETS[a["asset"]] if a.get("asset") else {},
    )
    manager._require_prim = lambda path: None
    manager._isaac = SimpleNamespace(
        add_reference_to_stage=lambda usd_path, prim_path: None,
        get_prim_at_path=lambda path: SimpleNamespace(),
        SingleXFormPrim=lambda path: SimpleNamespace(
            set_world_pose=lambda **pose: poses.append(pose)
        ),
        SingleArticulation=lambda prim_path, name: SimpleNamespace(),
    )
    manager.world = SimpleNamespace(scene=SimpleNamespace(add=lambda art: None))
    monkeypatch.setattr(
        sim_manager,
        "_reset_articulation_at_pose",
        lambda world, art, position, orientation: resets.append(
            (position, orientation)
        ),
    )
    monkeypatch.setattr(
        sim_manager, "IsaacArmHandle", lambda *args, **kwargs: SimpleNamespace()
    )
    manager._create_arm_isaac("arm", attrs)
    return poses[0], resets[0]


def test_ur20_spawns_turned_to_match_its_kinematics(monkeypatch):
    usd_pose, (position, orientation) = _spawn(
        monkeypatch, {"asset": "ur20", "position": [1.5, 0.0, 0.0]}
    )
    assert usd_pose["position"] == pytest.approx([1.5, 0.0, 0.0])
    assert _rotate(usd_pose["orientation"], (1, 0, 0)) == pytest.approx(
        (-1, 0, 0), abs=1e-9
    )
    assert orientation == pytest.approx(usd_pose["orientation"])
    assert position == pytest.approx((1.5, 0.0, 0.0))


def test_ur3e_spawn_is_unchanged(monkeypatch):
    usd_pose, (_, orientation) = _spawn(monkeypatch, {"asset": "ur3e"})
    assert "orientation" not in usd_pose
    assert orientation is None


def test_explicit_zero_offset_restores_the_raw_usd_orientation(monkeypatch):
    usd_pose, (_, orientation) = _spawn(
        monkeypatch, {"asset": "ur20", "base_yaw_offset_deg": 0.0}
    )
    assert "orientation" not in usd_pose
    assert orientation is None


def test_frame_orientation_composes_with_the_asset_offset(monkeypatch):
    frame = quat_from_euler_deg(0.0, 0.0, 90.0)
    usd_pose, _ = _spawn(
        monkeypatch, {"asset": "ur5e", "orientation_wxyz": list(frame)}
    )
    assert _rotate(usd_pose["orientation"], (1, 0, 0)) == pytest.approx(
        (0, -1, 0), abs=1e-9
    )


@pytest.mark.parametrize("value", (True, "180", float("inf"), float("nan")))
def test_base_yaw_offset_must_be_a_finite_number(value):
    config = ComponentConfig(
        name="arm",
        attributes=dict_to_struct(
            {"world": "sim-world", "asset": "ur20", "base_yaw_offset_deg": value}
        ),
    )

    with pytest.raises(ValueError, match="base_yaw_offset_deg"):
        IsaacArm.validate_config(config)
