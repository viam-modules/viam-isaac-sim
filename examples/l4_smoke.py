"""Headless smoke probe for the Lite 6 episode-generation cell."""
# ruff: noqa: ASYNC230

from __future__ import annotations

import asyncio
import argparse
import math
import json
import os
import threading
import traceback
from io import BytesIO

from PIL import Image
from viam.components.arm import JointPositions
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimManager

if __package__:
    from .scripted_pick import (
        PICK_BLOCK_DEPTH_M,
        PICK_BLOCK_HEIGHT_M,
        PICK_BLOCK_WIDTH_M,
        PICK_YAW_RANGE_DEG,
        sample_prop_pose,
    )
else:
    from scripted_pick import (
        PICK_BLOCK_DEPTH_M,
        PICK_BLOCK_HEIGHT_M,
        PICK_BLOCK_WIDTH_M,
        PICK_YAW_RANGE_DEG,
        sample_prop_pose,
    )


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


async def _exercise(mock: bool) -> dict:
    world = IsaacWorld.new(
        _config(
            "sim-world",
            {
                "mock": mock,
                "headless": True,
                "livestream": False,
                "physics_dt": 1.0 / 60.0,
                "rendering_dt": 1.0 / 60.0,
                "boot_timeout_sec": 300,
                "props": [
                    {
                        "name": "pick_block",
                        "type": "cube",
                        "position": [2.5, 0.0, 0.02],
                        "size": 0.04,
                        "scale": [1.0, 1.0, 1.0],
                        "color": [1.0, 0.0, 0.0],
                    }
                ],
            },
        ),
        {},
    )
    arm = IsaacArm.new(
        _config(
            "pick-arm",
            {
                "world": "sim-world",
                "asset": "lite6",
                "end_effector_prim": "/World/pick_arm/gripper/uflite_gripper_link",
            },
        ),
        {},
    )
    gripper = IsaacGripper.new(
        _config(
            "pick-gripper",
            {
                "world": "sim-world",
                "asset": "lite6_gripper",
                "arm": "pick-arm",
                "jaw_layout": "wide",
                "sensorless": True,
                "grab_timeout_sec": 10,
            },
        ),
        {},
    )
    fixed = IsaacCamera.new(
        _config(
            "fixed-cam",
            {
                "world": "sim-world",
                "position": [1.3, -1.3, 1.6],
                "target": [0.3, 0.0, 0.05],
                "width": 640,
                "height": 360,
            },
        ),
        {},
    )
    wrist = IsaacCamera.new(
        _config(
            "wrist-cam",
            {
                "world": "sim-world",
                "parent_prim": "/World/pick_arm/gripper/uflite_gripper_link",
                "local_position": [0.04, 0.0, 0.04],
                "local_orientation_rpy_deg": [180.0, -30.0, 0.0],
                "near_clip_m": 0.01,
                "width": 640,
                "height": 360,
            },
        ),
        {},
    )

    home = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
    await arm.move_to_joint_positions(JointPositions(values=home))
    moved = list(home)
    moved[0] = 5.0
    await arm.move_to_joint_positions(JointPositions(values=moved))
    await arm.move_to_joint_positions(JointPositions(values=home))
    await gripper.open()
    if not await gripper.grab():
        raise AssertionError("Lite6 close command was not accepted")
    gripper_input = (await gripper.get_current_inputs())[0]
    await gripper._wait_stopped()
    if not (await gripper.is_holding_something()).is_holding_something:
        raise AssertionError("sensorless Lite6 did not report its active grip")
    await gripper.open()
    if (await gripper.is_holding_something()).is_holding_something:
        raise AssertionError("sensorless Lite6 still reported holding after open")
    joints = (await arm.get_joint_positions()).values

    region_min = (0.25, -0.12, 0.0)
    region_max = (0.38, 0.12, 0.0)
    realized = []
    for seed in (0, 1, 42):
        position, orientation = sample_prop_pose(
            seed,
            region_min,
            region_max,
            PICK_YAW_RANGE_DEG,
            PICK_BLOCK_WIDTH_M,
            PICK_BLOCK_DEPTH_M,
            PICK_BLOCK_HEIGHT_M,
        )
        pose = await world.do_command(
            {
                "command": "set_prop_pose",
                "prop": "pick_block",
                "position": list(position),
                "orientation_wxyz": list(orientation),
            }
        )
        queried = await world.do_command(
            {"command": "get_prop_pose", "prop": "pick_block"}
        )
        measured = queried["pose"]
        position_error = max(
            abs(float(a) - float(b))
            for a, b in zip(measured["position"], pose["position"])
        )
        orientation_dot = abs(
            sum(
                float(a) * float(b)
                for a, b in zip(measured["orientation_wxyz"], pose["orientation_wxyz"])
            )
        )
        if position_error > 1e-5 or not math.isclose(
            orientation_dot, 1.0, abs_tol=1e-5
        ):
            raise AssertionError(
                f"pose query disagrees for seed {seed}: {queried} != {pose}"
            )
        realized.append(pose)
    await world.do_command({"command": "park_props", "props": ["pick_block"]})
    replay_position, replay_orientation = sample_prop_pose(
        42,
        region_min,
        region_max,
        PICK_YAW_RANGE_DEG,
        PICK_BLOCK_WIDTH_M,
        PICK_BLOCK_DEPTH_M,
        PICK_BLOCK_HEIGHT_M,
    )
    replay = await world.do_command(
        {
            "command": "set_prop_pose",
            "prop": "pick_block",
            "position": list(replay_position),
            "orientation_wxyz": list(replay_orientation),
        }
    )
    if replay != realized[-1]:
        raise AssertionError("same-seed prop placement did not replay")
    fixed_image, wrist_image = await asyncio.gather(
        fixed.get_image(), wrist.get_image()
    )
    fixed_size = Image.open(BytesIO(fixed_image.data)).size
    wrist_frame = Image.open(BytesIO(wrist_image.data)).convert("RGB")
    wrist_size = wrist_frame.size
    wrist_luma = wrist_frame.convert("L")
    histogram = wrist_luma.histogram()
    visible_fraction = sum(histogram[16:]) / (wrist_size[0] * wrist_size[1])
    if visible_fraction < 0.1:
        raise AssertionError(
            f"wrist camera frame is blank ({visible_fraction:.1%} visible pixels)"
        )
    if output_path := os.environ.get("L4_SMOKE_FIXED_FRAME"):
        with open(output_path, "wb") as output:
            output.write(fixed_image.data)
    if output_path := os.environ.get("L4_SMOKE_WRIST_FRAME"):
        with open(output_path, "wb") as output:
            output.write(wrist_image.data)
    return {
        "status": "PASS",
        "arm_joint_count": len(joints),
        "gripper_input": gripper_input,
        "fixed_camera": list(fixed_size),
        "wrist_camera": list(wrist_size),
        "target_seed_count": len(realized),
        "target_replay": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock", action="store_true", help="run without Isaac or a GPU"
    )
    args = parser.parse_args()
    sim = SimManager.get()
    result: dict = {}

    def client() -> None:
        nonlocal result
        try:
            result = asyncio.run(_exercise(args.mock))
        except BaseException as exc:
            result = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        finally:
            print(json.dumps(result, sort_keys=True), flush=True)
            # Kit's normal teardown can terminate or crash the interpreter.
            os._exit(0 if result.get("status") == "PASS" else 1)

    worker = threading.Thread(target=client, name="l4-smoke-client")
    worker.start()
    sim.main_loop()


if __name__ == "__main__":
    raise SystemExit(main())
