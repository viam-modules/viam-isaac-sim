"""End-to-end test of the module in mock mode: exercises the viam component
models against the in-process mock backend."""

import asyncio
import math

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Vector3
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm, _densify
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.world import IsaacWorld


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture(scope="module")
def world():
    return IsaacWorld.new(_config("sim-world", {"mock": True}), {})


def test_world_configures_and_status(world):
    status = asyncio.run(world.do_command({"command": "status"}))
    assert status["configured"] is True
    assert status["mock"] is True


def test_validate_requires_world():
    with pytest.raises(ValueError, match="world"):
        IsaacArm.validate_config(_config("a", {"asset": "ur20"}))


def test_validate_requires_source():
    with pytest.raises(ValueError, match="asset"):
        IsaacArm.validate_config(_config("a", {"world": "sim-world"}))


def test_validate_ok_returns_dependency():
    deps, _ = IsaacArm.validate_config(
        _config("a", {"world": "sim-world", "asset": "ur20"})
    )
    assert list(deps) == ["sim-world"]


def test_arm_moves(world):
    arm = IsaacArm.new(
        _config("my-arm", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {}
    )

    async def scenario():
        start = await arm.get_joint_positions()
        assert start.values == pytest.approx([0.0] * 6)

        from viam.components.arm import JointPositions

        target = JointPositions(values=[10, -20, 30, 0, 5, -5])
        await arm.move_to_joint_positions(target)
        assert not await arm.is_moving()
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)

        pose = await arm.get_end_position()
        assert pose.x == pytest.approx(300.0)
        assert pose.o_z == pytest.approx(1.0)

        # trajectory execution (what the motion service calls)
        waypoints = [
            JointPositions(values=[5, -10, 15, 0, 2, -2]),
            JointPositions(values=[0, 0, 0, 0, 0, 0]),
        ]
        await arm.move_through_joint_positions(waypoints)
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([0] * 6, abs=0.5)

    asyncio.run(scenario())


def test_densify_pass_through_dense_input():
    start = [0.0] * 6
    wps = [[math.radians(0.5 * i)] * 6 for i in range(1, 5)]
    out = _densify(start, wps, math.radians(2.0))
    assert out == wps


def test_densify_interpolates_sparse_input():
    start = [0.0] * 6
    wps = [[math.radians(30)] + [0.0] * 5]
    out = _densify(start, wps, math.radians(2.0))
    assert len(out) == 15  # 30° / 2°
    prev = start
    for step in out:
        max_delta = max(abs(a - b) for a, b in zip(prev, step))
        assert max_delta <= math.radians(2.0) + 1e-12
        prev = step
    assert out[-1] == pytest.approx(wps[-1])


def test_densify_returns_final_target_exactly():
    start = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    wps = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.7, 0.0, 0.0, 0.0, 0.0, 0.0]]
    out = _densify(start, wps, math.radians(3.0))
    assert out[-1] == pytest.approx(wps[-1])


def test_densify_empty_input_returns_empty():
    assert _densify([0.0] * 6, [], math.radians(2.0)) == []


def test_arm_reconfigure_rejects_invalid_max_step_deg(world):
    with pytest.raises(ValueError, match="path_max_step_deg"):
        IsaacArm.new(
            _config("bad", {"world": "sim-world", "asset": "ur20", "path_max_step_deg": 0})
            , {},
        )


def test_arm_reconfigure_rejects_invalid_control_freq(world):
    with pytest.raises(ValueError, match="robot_control_freq_hz"):
        IsaacArm.new(
            _config("bad", {"world": "sim-world", "asset": "ur20", "robot_control_freq_hz": -1})
            , {},
        )


def test_arm_move_through_sparse_waypoints_reaches_final_target(world):
    arm = IsaacArm.new(
        _config(
            "sparse-arm",
            {
                "world": "sim-world",
                "asset": "ur20",
                "mock_dof": 6,
                "path_max_step_deg": 2.0,
                "robot_control_freq_hz": 200,
                "path_tolerance_delta_deg": 1.0,
                "move_timeout_sec": 10,
            },
        ),
        {},
    )

    from viam.components.arm import JointPositions

    # Non-colinear polyline: shortest path from home to final skips both
    # intermediates, so reaching final within tolerance only proves the drives
    # tracked the streamed intermediates.
    waypoints = [
        JointPositions(values=[20, 0, 0, 0, 0, 0]),
        JointPositions(values=[-20, 0, 0, 0, 0, 0]),
        JointPositions(values=[0, 0, 0, 0, 0, 0]),
    ]

    async def scenario():
        await arm.move_through_joint_positions(waypoints)
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([0] * 6, abs=1.0)

    asyncio.run(scenario())


def test_camera_returns_image(world):
    cam = IsaacCamera.new(
        _config("my-cam", {"world": "sim-world", "width": 320, "height": 240}), {}
    )

    async def scenario():
        img = await cam.get_image()
        assert img.mime_type == "image/jpeg"
        assert len(img.data) > 100

        from PIL import Image
        from io import BytesIO

        decoded = Image.open(BytesIO(img.data))
        assert decoded.size == (320, 240)

        images, _meta = await cam.get_images()
        assert len(images) == 1

        props = await cam.get_properties()
        assert props.supports_pcd is False

    asyncio.run(scenario())


def test_base_drives(world):
    base = IsaacBase.new(
        _config("my-base", {"world": "sim-world", "asset": "jetbot"}), {}
    )

    async def scenario():
        assert not await base.is_moving()
        await base.set_velocity(Vector3(x=0, y=200, z=0), Vector3(x=0, y=0, z=45))
        assert await base.is_moving()
        await base.stop()
        assert not await base.is_moving()

        # short timed move
        await base.move_straight(distance=10, velocity=100)
        assert not await base.is_moving()

        props = await base.get_properties()
        assert props.wheel_circumference_meters == pytest.approx(2 * math.pi * 0.05)

    asyncio.run(scenario())
