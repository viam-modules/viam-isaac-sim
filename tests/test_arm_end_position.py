"""GetEndPosition reports the end effector in the arm's base frame, like a
real arm driver; episode capture keeps recording it in world coordinates."""

from types import SimpleNamespace

import pytest

from isaac_module import sim_manager
from isaac_module.sim_manager import KNOWN_ASSETS, IsaacArmHandle, SimManager
from isaac_module.spatial import quat_from_euler_deg

IDENTITY = (1.0, 0.0, 0.0, 0.0)


class _FakeSim:
    def run(self, fn, timeout=None):
        return fn()


def _handle(ee_position, ee_rotation=IDENTITY, **base):
    art = SimpleNamespace(dof_names=list(range(6)))
    ee = SimpleNamespace(get_world_pose=lambda: (ee_position, ee_rotation))
    return IsaacArmHandle(_FakeSim(), art, ee, "/World/arm", **base)


def test_arm_at_the_origin_reports_the_world_pose():
    yaw = quat_from_euler_deg(0.0, 0.0, 30.0)
    position, rotation = _handle((0.4, 0.1, 0.6), yaw).get_end_pose_in_base()
    assert position == pytest.approx((0.4, 0.1, 0.6))
    assert rotation == pytest.approx(yaw)


def test_offset_arm_reports_relative_to_its_base():
    handle = _handle((2.0, 0.25, 0.5), base_position=(1.5, 0.0, 0.0))
    position, rotation = handle.get_end_pose_in_base()
    assert position == pytest.approx((0.5, 0.25, 0.5))
    assert rotation == pytest.approx(IDENTITY)
    # episode capture reads get_end_pose, which stays in world coordinates
    assert handle.get_end_pose()[0] == pytest.approx((2.0, 0.25, 0.5))


def test_rotated_base_rotates_the_reported_pose():
    yaw = quat_from_euler_deg(0.0, 0.0, 90.0)
    handle = _handle(
        (1.0, 0.5, 0.2), yaw, base_position=(1.0, 0.0, 0.0), base_orientation=yaw
    )
    position, rotation = handle.get_end_pose_in_base()
    assert position == pytest.approx((0.5, 0.0, 0.2), abs=1e-9)
    assert rotation == pytest.approx(IDENTITY, abs=1e-9)


def test_spawn_gives_the_handle_the_frame_pose_without_the_usd_yaw_offset(
    monkeypatch,
):
    handle_kwargs = {}
    manager = object.__new__(SimManager)
    manager._arm_articulations = {}
    manager._arm_usd_owners = {}
    manager._shared_bases = []
    manager._resolve_usd = lambda a: ("/fake/arm.usd", KNOWN_ASSETS[a["asset"]])
    manager._require_prim = lambda path: None
    manager._isaac = SimpleNamespace(
        add_reference_to_stage=lambda usd_path, prim_path: None,
        get_prim_at_path=lambda path: SimpleNamespace(),
        SingleXFormPrim=lambda path: SimpleNamespace(set_world_pose=lambda **_: None),
        SingleArticulation=lambda prim_path, name: SimpleNamespace(),
    )
    manager.world = SimpleNamespace(scene=SimpleNamespace(add=lambda art: None))
    monkeypatch.setattr(sim_manager, "_reset_articulation_at_pose", lambda *a: None)
    monkeypatch.setattr(
        sim_manager,
        "IsaacArmHandle",
        lambda *args, **kwargs: handle_kwargs.update(kwargs),
    )
    frame = quat_from_euler_deg(0.0, 0.0, 90.0)
    manager._create_arm_isaac(
        "arm-2",
        {"asset": "ur20", "position": [1.5, 0.0, 0.0], "orientation_wxyz": list(frame)},
    )
    assert handle_kwargs["base_position"] == pytest.approx((1.5, 0.0, 0.0))
    assert handle_kwargs["base_orientation"] == pytest.approx(list(frame))
