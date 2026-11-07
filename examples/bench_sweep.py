"""Deterministic sim-loop benchmark: sweep the arm base yaw with both cameras recording.

No grasping, no world reset, no home pose. Time-based: latches targets and holds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time

from viam.components.arm import Arm
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


def _err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


async def _latch(arm, deg: list[float]) -> None:
    await arm.do_command(
        {"command": "set_joint_targets_radians", "values": [math.radians(d) for d in deg]}
    )


async def _run(args: argparse.Namespace, world, arm) -> dict:
    # Fixed start pose (straight up, wrist unfolded); the settle is not measured.
    start = [0.0] * 6
    await _latch(arm, start)
    await asyncio.sleep(args.settle_sec)
    before = await world.do_command({"command": "status"})
    started = time.monotonic()
    wps = sweep_waypoints(start, args.amplitude_deg, args.cycles)
    error = None
    try:
        pose = (await world.do_command({"command": "get_prop_pose", "prop": args.prop}))[
            "pose"
        ]
        await world.do_command(
            {
                "command": "start_episode",
                "output_dir": args.output_dir,
                "episode_id": str(int(time.time())),
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
                await _latch(arm, wp)
                await asyncio.sleep(args.hold_sec)
        finally:
            try:
                await world.do_command(
                    {"command": "stop_episode", "outcome": "incomplete"}
                )
            except Exception as exc:
                error = error or _err(exc)
    except Exception as exc:
        error = _err(exc)
    wall_s = time.monotonic() - started
    perf = None
    try:
        after = await world.do_command({"command": "status"})
        perf = perf_report(before, after, wall_s)
    except Exception as exc:
        error = error or _err(exc)
    return {
        "perf": perf,
        "waypoints": len(wps),
        "start_joints_deg": start,
        "hold_sec": args.hold_sec,
        "error": error,
    }


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
    if args.disable_webrtc:
        assert options.dial_options is not None
        options.dial_options.disable_webrtc = True
    async with await RobotClient.at_address(args.address, options) as machine:
        return await _run(
            args,
            Generic.from_robot(machine, args.world),
            Arm.from_robot(machine, args.arm),
        )


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
    p.add_argument("--settle-sec", type=float, default=5.0)
    p.add_argument("--hold-sec", type=float, default=3.0)
    p.add_argument("--disable-webrtc", action="store_true")
    args = p.parse_args()
    try:
        result = asyncio.run(_main(args))
    except Exception as exc:
        result = {"perf": None, "waypoints": 0, "error": _err(exc)}
    print(json.dumps(result))
    return 1 if result["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
