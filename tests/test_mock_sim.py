"""End-to-end test of the module in mock mode: boots the SimManager on a
background thread (standing in for the process main thread) and exercises
the viam component models against it."""

import asyncio
import math
import threading
from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Vector3
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm, _densify
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import (
    SimConfig,
    SimManager,
    _visual_material_properties,
    default_rendering_dt,
    validate_rendering_dt,
)


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture(scope="module")
def sim():
    previous = SimManager._instance
    manager = SimManager()
    SimManager._instance = manager
    t = threading.Thread(target=manager.main_loop, daemon=True)
    t.start()
    yield manager
    manager.request_stop()
    t.join(timeout=5)
    SimManager._instance = previous


@pytest.fixture(scope="module")
def world(sim):
    return IsaacWorld.new(
        _config(
            "sim-world",
            {
                "mock": True,
                "props": [
                    {
                        "name": "pick_block",
                        "type": "cube",
                        "position": [2.5, 0.0, 0.0125],
                        "size": 0.075,
                        "scale": [1.0, 0.2, 1.0 / 3.0],
                    }
                ],
            },
        ),
        {},
    )


def test_world_boots_and_status(world):
    status = asyncio.run(world.do_command({"command": "status"}))
    assert status["booted"] is True
    assert status["mock"] is True


def test_mock_sim_clock_advances(sim):
    import time

    before = sim.sim_time_s()
    time.sleep(0.05)
    assert sim.sim_time_s() > before


@pytest.mark.parametrize(
    ("physics_dt", "capture_rate_hz", "expected"),
    (
        (1 / 60, 20.0, 1 / 20),
        (1 / 60, 15.0, 1 / 15),
        (1 / 60, 60.0, 1 / 60),
        (1 / 100, 20.0, 1 / 20),
        # 7 Hz is not a whole number of 1/60 steps: render every physics step
        (1 / 60, 7.0, 1 / 60),
        # capture faster than physics: render every physics step
        (1 / 20, 60.0, 1 / 20),
    ),
)
def test_default_rendering_dt_renders_once_per_capture_sample(
    physics_dt, capture_rate_hz, expected
):
    assert default_rendering_dt(physics_dt, capture_rate_hz) == pytest.approx(
        expected
    )


def test_sim_config_resolves_rendering_dt():
    assert SimConfig(capture_rate_hz=20.0).resolved_rendering_dt() == pytest.approx(
        1 / 20
    )
    assert SimConfig(rendering_dt=1 / 60).resolved_rendering_dt() == pytest.approx(
        1 / 60
    )
    assert SimConfig(
        physics_dt=1 / 120, rendering_dt=1 / 30
    ).resolved_rendering_dt() == pytest.approx(1 / 30)
    with pytest.raises(ValueError, match="integer multiple"):
        SimConfig(physics_dt=1 / 60, rendering_dt=1 / 40).resolved_rendering_dt()
    with pytest.raises(ValueError, match="integer multiple"):
        SimConfig(physics_dt=1 / 60, rendering_dt=1 / 120).resolved_rendering_dt()


def test_world_validate_config_rejects_misaligned_rendering_dt():
    with pytest.raises(ValueError, match="integer multiple"):
        IsaacWorld.validate_config(
            _config("w", {"physics_dt": 1 / 60, "rendering_dt": 1 / 40})
        )
    IsaacWorld.validate_config(_config("w", {"rendering_dt": 1 / 20}))
    validate_rendering_dt(1 / 60, 1 / 20)


def test_visual_material_presets_are_validated_before_boot():
    assert _visual_material_properties("wood")["metallic"] == 0.0
    assert _visual_material_properties("brushed_metal")["metallic"] == 1.0
    manager = SimManager()
    manager._register_props([{"name": "table", "visual_material": "wood"}])
    assert manager._prop_specs["table"]["visual_material"] == "wood"
    with pytest.raises(ValueError, match="visual_material"):
        manager._register_props([{"name": "table", "visual_material": "marble"}])

def test_spawned_cubes_override_isaac_contact_offsets():
    offsets = []

    class FakeCuboid:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_rest_offset(self, value):
            offsets.append(("rest", value))

        def set_contact_offset(self, value):
            offsets.append(("contact", value))

    manager = object.__new__(SimManager)
    manager._isaac = SimpleNamespace(
        PhysicsMaterial=lambda **kwargs: object(),
        DynamicCuboid=FakeCuboid,
        FixedCuboid=FakeCuboid,
    )
    added = []
    manager.world = SimpleNamespace(scene=SimpleNamespace(add=added.append))

    manager._spawn_prop({"name": "block", "type": "cube"})

    assert offsets == [("rest", 0.0), ("contact", 0.001)]
    assert len(added) == 1


def test_world_park_props_restores_configured_pose(world):
    asyncio.run(
        world.do_command(
            {
                "command": "set_prop_pose",
                "prop": "pick_block",
                "position": [0.5, 0.1, 0.025],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        )
    )

    parked = asyncio.run(
        world.do_command({"command": "park_props", "props": ["pick_block"]})
    )

    assert parked["poses"]["pick_block"]["position"] == [2.5, 0.0, 0.0125]


def test_world_set_prop_pose_accepts_and_returns_posed(world):
    """Test that set_prop_pose command applies and returns the given pose."""
    # Set prop to a non-default position and quaternion (45 degree rotation)
    position = [0.5, 0.1, 0.025]
    orientation_wxyz = [math.cos(math.radians(45) / 2), 0, 0, math.sin(math.radians(45) / 2)]

    result = asyncio.run(
        world.do_command(
            {
                "command": "set_prop_pose",
                "prop": "pick_block",
                "position": position,
                "orientation_wxyz": orientation_wxyz,
            }
        )
    )

    # Assert returned pose matches what was set
    assert result["position"] == pytest.approx(position)
    assert result["orientation_wxyz"] == pytest.approx(orientation_wxyz)

    # Query the pose and verify it matches
    queried = asyncio.run(
        world.do_command({"command": "get_prop_pose", "prop": "pick_block"})
    )
    assert queried["pose"]["position"] == pytest.approx(position)
    assert queried["pose"]["orientation_wxyz"] == pytest.approx(orientation_wxyz)



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
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)
        # the move returns inside the 0.5 degree tolerance; the mock covers that
        # last window at 1 rad/s in under 10 ms and then snaps to its target
        await asyncio.sleep(0.05)
        assert not await arm.is_moving()

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


def test_arm_set_joint_targets_radians_is_immediate_and_validated(world):
    arm = IsaacArm.new(
        _config("stream-arm", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}),
        {},
    )

    async def scenario():
        target = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
        result = await arm.do_command(
            {"command": "set_joint_targets_radians", "values": target}
        )
        assert result == {"accepted": True}
        # Component is_moving tracks MoveTo RPCs; the raw handle tracks drive motion.
        assert not await arm.is_moving()
        assert arm._h().is_moving()
        await arm.stop()
        with pytest.raises(ValueError, match="six values"):
            await arm.do_command(
                {"command": "set_joint_targets_radians", "values": [0.0]}
            )

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

        from io import BytesIO

        from PIL import Image

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


def test_capture_after_render_samples_once_per_capture_period():
    from types import SimpleNamespace

    from isaac_module.sim_manager import SimConfig, SimManager

    manager = SimManager()
    manager.cfg = SimConfig(capture_rate_hz=15.0, capture_after_render=True)
    manager._episode_writer = SimpleNamespace(failed=False)
    sampled = []
    manager._capture_sample = lambda sim_time_s: (
        sampled.append(round(sim_time_s, 6)),
        setattr(manager, "_capture_frame_index", manager._capture_frame_index + 1),
    )
    manager.world = SimpleNamespace(current_time=10.0)
    manager._capture_started_sim_s = 10.0
    manager._capture_last_sim_s = 10.0

    # Rendering once per capture period samples after every update.
    for _ in range(3):
        manager.world.current_time += 1.0 / 15.0
        manager._capture_after_render()
    assert sampled == [round(10.0 + k / 15.0, 6) for k in range(3)]

    # Rendering every physics step samples after every fourth update, and a
    # paused sim (no sim time passing) samples nothing.
    for _ in range(8):
        manager.world.current_time += 1.0 / 60.0
        manager._capture_after_render()
    manager._capture_after_render()
    assert len(sampled) == 5
