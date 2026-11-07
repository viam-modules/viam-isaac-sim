"""Deterministic sim-loop benchmark: sweep the arm base yaw with both cameras recording.

No grasping, no world reset, no home pose.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

if __package__:
    from .batch_picks import perf_report
else:
    from batch_picks import perf_report


def sweep_waypoints(
    current: list[float], amplitude_deg: float, cycles: int
) -> list[list[float]]:
    """Only joint 0 (base yaw) moves: +/-amplitude per cycle, then back to start."""
    yaws = [current[0] + s * amplitude_deg for _ in range(cycles) for s in (1, -1)]
    return [[yaw, *current[1:]] for yaw in [*yaws, current[0]]]


async def _main(args: argparse.Namespace) -> dict:
    api_key = os.environ.get("VIAM_API_KEY")
    api_key_id = os.environ.get("VIAM_API_KEY_ID")
    if bool(api_key) != bool(api_key_id):
        raise ValueError("set both VIAM_API_KEY and VIAM_API_KEY_ID, or neither")
    options = (
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
        if api_key and api_key_id
        else RobotClient.Options(dial_options=DialOptions(insecure=True))
    )
    async with await RobotClient.at_address(args.address, options) as machine:
        world = Generic.from_robot(machine, args.world)
        arm = Arm.from_robot(machine, args.arm)
        before = await world.do_command({"command": "status"})
        started = time.monotonic()
        wps: list[list[float]] = []
        error = None
        try:
            current = list((await arm.get_joint_positions()).values)
            wps = sweep_waypoints(current, args.amplitude_deg, args.cycles)
            pose = (
                await world.do_command({"command": "get_prop_pose", "prop": args.prop})
            )["pose"]
            await world.do_command(
                {
                    "command": "start_episode",
                    "output_dir": args.output_dir,
                    "episode_id": "000000",
                    "seed": 0,
                    "task": "perf sweep",
                    "arm": args.arm,
                    "gripper": args.gripper,
                    "fixed_camera": args.fixed_camera,
                    "wrist_camera": args.wrist_camera,
                    "target": {
                        "prop_name": args.prop,
                        "position_m": pose["position"],
                        "rotation_wxyz": pose["orientation_wxyz"],
                    },
                }
            )
            try:
                for wp in wps:
                    await arm.move_to_joint_positions(JointPositions(values=wp))
            finally:
                try:
                    await world.do_command(
                        {"command": "stop_episode", "outcome": "incomplete"}
                    )
                except Exception as exc:  # report, don't mask an earlier error
                    print(f"stop_episode failed: {exc}", file=sys.stderr)
        except Exception as exc:
            error = str(exc)
        after = await world.do_command({"command": "status"})
        wall_s = time.monotonic() - started
    return {
        "perf": perf_report(before, after, wall_s),
        "waypoints": len(wps),
        "error": error,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("output_dir")
    p.add_argument("--cycles", type=int, default=4)
    p.add_argument("--amplitude-deg", type=float, default=60.0)
    p.add_argument("--world", default="sim-world")
    p.add_argument("--arm", default="pick-arm")
    p.add_argument("--gripper", default="pick-gripper")
    p.add_argument("--fixed-camera", default="scene-cam")
    p.add_argument("--wrist-camera", default="wrist-cam")
    p.add_argument("--prop", default="pick_block")
    result = asyncio.run(_main(p.parse_args()))
    print(json.dumps(result))
    return 1 if result["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
