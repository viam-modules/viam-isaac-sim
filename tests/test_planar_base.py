"""Holonomic bases over an articulation's x/y/yaw joints, and prismatic joint units."""

import math
from types import SimpleNamespace

import pytest

from isaac_module import sim_manager
from isaac_module.models.arm import _scales
from isaac_module.sim_manager import SimManager, planar_step

RIDGEBACK = ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"]


def test_planar_step_moves_in_the_base_frame():
    # facing +y: forward is world +y, left is world -x
    x, y, yaw = planar_step([1.0, 2.0, math.pi / 2], forward=0.5, left=0.25, yaw_rate=0.1, dt=2.0)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(3.0)
    assert yaw == pytest.approx(math.pi / 2 + 0.2)


def test_prismatic_joints_report_millimeters():
    handle = SimpleNamespace(prismatic=(True, True, False))
    assert _scales(handle, 3) == [1000.0, 1000.0, pytest.approx(180 / math.pi)]
    assert _scales(SimpleNamespace(), 2) == [pytest.approx(180 / math.pi)] * 2


def test_prismatic_joint_names_come_from_the_usd():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.PrismaticJoint.Define(stage, "/robot/dummy_base_prismatic_x_joint")
    UsdPhysics.RevoluteJoint.Define(stage, "/robot/dummy_base_revolute_z_joint")
    assert sim_manager._prismatic_joint_names(stage.GetPrimAtPath("/robot")) == {
        "dummy_base_prismatic_x_joint"
    }


def test_planar_base_needs_the_arm_that_loaded_its_articulation():
    manager = object.__new__(SimManager)
    manager._arm_articulations = {}
    with pytest.raises(ValueError, match="depends_on"):
        manager._create_planar_base_isaac(
            "base", {"prim_path": "/World/robot", "planar_joints": RIDGEBACK}
        )

    callbacks = {}
    manager._arm_articulations["/World/robot"] = SimpleNamespace(
        dof_names=["panda_joint1", *RIDGEBACK]
    )
    manager.world = SimpleNamespace(add_physics_callback=lambda n, f: callbacks.update({n: f}))
    handle = manager._create_planar_base_isaac(
        "base", {"prim_path": "/World/robot", "planar_joints": RIDGEBACK}
    )
    assert handle._indices == [1, 2, 3]
    assert "base_drive" in callbacks
    with pytest.raises(ValueError, match="x, y, yaw"):
        manager._create_planar_base_isaac(
            "b2", {"prim_path": "/World/robot", "planar_joints": RIDGEBACK[:2]}
        )
