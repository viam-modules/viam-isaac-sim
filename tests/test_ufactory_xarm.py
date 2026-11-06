"""The ufactory-xarm6 model takes viam:ufactory:xArm6 configs unchanged."""

import asyncio
import threading

import pytest
from viam.components.arm import JointPositions
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.ufactory_xarm import UfactoryXArm6, isaac_arm_attrs
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import KNOWN_ASSETS, SimManager

# a real xArm 6 config, plus the world attribute the app adds when simulating
REAL_ATTRS = {
    "host": "10.1.1.26",
    "speed_degs_per_sec": 90,
    "acceleration_degs_per_sec_per_sec": 200,
    "ufactory-studio-proxy": True,
    "collision_sensitivity": 3,
}


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture(scope="module")
def world():
    previous = SimManager._instance
    manager = SimManager()
    SimManager._instance = manager
    t = threading.Thread(target=manager.main_loop, daemon=True)
    t.start()
    yield IsaacWorld.new(_config("sim-world", {"mock": True}), {})
    manager.request_stop()
    t.join(timeout=5)
    SimManager._instance = previous


def test_validate_requires_world():
    with pytest.raises(ValueError, match="world"):
        UfactoryXArm6.validate_config(_config("arm", REAL_ATTRS))


def test_validate_returns_world_and_optional_motion():
    deps, optional = UfactoryXArm6.validate_config(
        _config("arm", {**REAL_ATTRS, "world": "sim-world"})
    )
    assert list(deps) == ["sim-world"]
    assert list(optional) == ["rdk:service:motion/builtin"]


def test_validate_requires_configured_motion():
    deps, optional = UfactoryXArm6.validate_config(
        _config("arm", {**REAL_ATTRS, "world": "sim-world", "motion": "planner"})
    )
    assert list(deps) == ["sim-world", "rdk:service:motion/planner"]
    assert list(optional) == []


@pytest.mark.parametrize(
    "attrs, message",
    [
        ({"speed_degs_per_sec": 200}, "speed"),
        ({"speed_degs_per_sec": 0.1}, "speed"),
        ({"acceleration_degs_per_sec_per_sec": 2000}, "acceleration"),
        ({"acceleration_degs_per_sec_per_sec": -1}, "acceleration"),
        ({"move_hz": 5}, "move_hz"),
    ],
)
def test_validate_rejects_what_the_real_driver_rejects(attrs, message):
    with pytest.raises(ValueError, match=message):
        UfactoryXArm6.validate_config(_config("arm", {"world": "sim-world", **attrs}))


def test_attrs_follow_the_real_driver_defaults():
    attrs = isaac_arm_attrs({"world": "sim-world", "host": "10.1.1.26"}, "arm")
    assert attrs == {
        "world": "sim-world",
        "asset": "xarm6",
        "max_joint_deg_s": 60.0,
        "max_joint_deg_s2": pytest.approx(1145.0 / 3),
        "robot_control_freq_hz": 100.0,
        "path_max_step_deg": pytest.approx(0.6),
    }


def test_attrs_carry_configured_limits():
    attrs = isaac_arm_attrs({**REAL_ATTRS, "world": "sim-world", "move_hz": 50}, "arm")
    assert attrs["max_joint_deg_s"] == 90.0
    assert attrs["max_joint_deg_s2"] == 200.0
    assert attrs["robot_control_freq_hz"] == 50.0
    assert attrs["path_max_step_deg"] == pytest.approx(1.8)


def test_xarm6_asset_drives_only_the_arm_joints():
    asset = KNOWN_ASSETS["xarm6"]
    assert asset["usd"] == ["/Isaac/Robots/Ufactory/xarm6/xarm6.usd"]
    assert asset["joint_names"] == tuple(f"joint{i}" for i in range(1, 7))
    assert asset["kinematics"].endswith("/xarm6_kinematics.json")


def test_arm_moves(world):
    arm = UfactoryXArm6.new(_config("xarm", {**REAL_ATTRS, "world": "sim-world"}), {})

    async def scenario():
        await arm.move_to_joint_positions(
            JointPositions(values=[10, -20, 30, 0, 5, -5])
        )
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)

        await arm.move_through_joint_positions([JointPositions(values=[0] * 6)])
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([0] * 6, abs=0.5)

        with pytest.raises(ValueError, match="motion service"):
            await arm.move_to_position(await arm.get_end_position())

    asyncio.run(scenario())
