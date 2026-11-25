"""Check lockstep replay on a running isaac-sim machine.

    VIAM_API_KEY_ID=... VIAM_API_KEY=... python examples/replay_lockstep_check.py \
        <machine-address> --arm arm --prop cup [--gripper g --fixed-camera c1 \
        --wrist-camera c2 --output-dir runs/lockstep]

Runs, and prints JSON for, each check:
  step       paused `step`: each sim_time_s delta is ticks x rendering_dt, and
             `step` on a playing world errors
  teleport   set_joint_positions_radians 25 deg away, then 20 single steps:
             max joint deviation from the teleported pose
  throughput paused replay of --states states, two ways: "two_call" (teleport
             + set prop gathered, then step) and "combined" (one step
             carrying arms and props): states/s, median and p95 per state
  concurrent three gathered get_prop_pose calls on a playing world take at
             most two ticks more than one call (not serialized to three)
  episode    (needs --gripper and cameras) paused replay of 41 states into
             one native episode, one combined step per state: frame count
Leaves the world playing.
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import time

from viam.components.arm import Arm
from viam.components.generic import Generic
from viam.robot.client import RobotClient


def _state(home: list[float], k: int) -> tuple[list[float], list[float]]:
    """Recorded state k: arm joints (rad), prop position (m)."""
    t = k / 20.0
    q = list(home)
    q[0] += math.radians(10.0) * math.sin(2 * math.pi * 0.25 * t)
    return q, [0.45 + 0.05 * math.cos(t), 0.05 * math.sin(t), 0.05]


async def _set(arm, world, prop, q, position):
    await asyncio.gather(
        arm.do_command({"command": "set_joint_positions_radians", "values": q}),
        world.do_command(
            {
                "command": "set_prop_pose",
                "prop": prop,
                "position": position,
                "orientation_wxyz": [1, 0, 0, 0],
            }
        ),
    )


async def _joints(arm) -> list[float]:
    return list((await arm.do_command({"command": "get_joint_positions_radians"}))["values"])


async def check_step(world, rendering_dt):
    await world.do_command({"command": "play"})
    try:
        await world.do_command({"command": "step"})
        errors_while_playing = False
    except Exception as exc:
        errors_while_playing = "paused" in str(exc)
    await world.do_command({"command": "pause"})
    last = float((await world.do_command({"command": "status"}))["sim_time"])
    deltas = []
    for ticks in [1] * 10 + [5]:
        result = await world.do_command({"command": "step", "ticks": ticks})
        now = float(result["sim_time_s"])
        deltas.append({"ticks": ticks, "delta_s": now - last})
        last = now
    worst = max(abs(d["delta_s"] - d["ticks"] * rendering_dt) for d in deltas)
    return {
        "pass": errors_while_playing and worst < 1e-6,
        "errors_while_playing": errors_while_playing,
        "max_error_s": worst,
        "deltas": deltas,
    }


async def check_teleport(arm, world):
    await world.do_command({"command": "pause"})
    q = await _joints(arm)
    q[0] += math.radians(25.0)
    await arm.do_command({"command": "set_joint_positions_radians", "values": q})
    deviation = []
    for _ in range(20):
        await world.do_command({"command": "step"})
        got = await _joints(arm)
        deviation.append(max(abs(a - b) for a, b in zip(got, q)))
    return {
        "pass": max(deviation) < math.radians(0.5),
        "max_deviation_deg": math.degrees(max(deviation)),
        "per_tick_deg": [round(math.degrees(d), 4) for d in deviation],
    }


def _step_with(arm_name, prop, q, position):
    return {
        "command": "step",
        "arms": {arm_name: q},
        "props": {prop: {"position": position, "orientation_wxyz": [1, 0, 0, 0]}},
    }


async def check_throughput(arm, world, prop, home, states):
    await world.do_command({"command": "pause"})

    async def two_call(k):
        await _set(arm, world, prop, *_state(home, k))
        await world.do_command({"command": "step"})

    async def combined(k):
        await world.do_command(_step_with(arm.name, prop, *_state(home, k)))

    report = {"baseline_states_per_s": 12.8}
    for name, replay in (("two_call", two_call), ("combined", combined)):
        per_state = []
        start = time.perf_counter()
        for k in range(states):
            t0 = time.perf_counter()
            await replay(k)
            per_state.append(time.perf_counter() - t0)
        wall = time.perf_counter() - start
        report[name] = {
            "states_per_s": states / wall,
            "median_s": statistics.median(per_state),
            "p95_s": sorted(per_state)[int(0.95 * states)],
        }
    return report


async def check_concurrent(world, prop, rendering_dt):
    await world.do_command({"command": "play"})
    one = []
    for _ in range(5):
        t0 = time.perf_counter()
        await world.do_command({"command": "get_prop_pose", "prop": prop})
        one.append(time.perf_counter() - t0)
    three = []
    for _ in range(5):
        t0 = time.perf_counter()
        await asyncio.gather(
            *(world.do_command({"command": "get_prop_pose", "prop": prop}) for _ in range(3))
        )
        three.append(time.perf_counter() - t0)
    return {
        "single_median_s": statistics.median(one),
        "three_gathered_median_s": statistics.median(three),
        # serialized, three calls would take three ticks
        "pass": statistics.median(three)
        <= statistics.median(one) + 2 * rendering_dt,
    }


async def check_episode(arm, world, prop, home, a):
    await world.do_command({"command": "pause"})
    await _set(arm, world, prop, *_state(home, 0))
    await world.do_command(
        {
            "command": "start_episode",
            "output_dir": a.output_dir,
            "episode_id": f"lockstep-{int(time.time())}",
            "task": "lockstep replay check",
            "arm": a.arm,
            "gripper": a.gripper,
            "fixed_camera": a.fixed_camera,
            "wrist_camera": a.wrist_camera,
            "target": {
                "prop_name": prop,
                "position_m": _state(home, 0)[1],
                "rotation_wxyz": [1, 0, 0, 0],
            },
        }
    )
    frames = []
    for k in range(1, 41):
        result = await world.do_command(_step_with(arm.name, prop, *_state(home, k)))
        frames.append(result["episode_frames"])
    stopped = await world.do_command({"command": "stop_episode", "outcome": "success"})
    return {
        "pass": stopped.get("sample_count") == 41 and frames == list(range(2, 42)),
        "sample_count": stopped.get("sample_count"),
        "path": stopped.get("path"),
        "frames_after_each_step": frames,
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("address")
    p.add_argument("--world", default="world")
    p.add_argument("--arm", default="arm")
    p.add_argument("--prop", default="cup")
    p.add_argument("--states", type=int, default=200)
    p.add_argument("--rendering-dt", type=float, default=0.05,
                   help="the world's rendering_dt (status doesn't report it)")
    p.add_argument("--gripper")
    p.add_argument("--fixed-camera", default="overhead-camera")
    p.add_argument("--wrist-camera", default="wrist-camera")
    p.add_argument("--output-dir", default="runs/lockstep")
    a = p.parse_args()
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"]
    )
    robot = await RobotClient.at_address(a.address, opts)
    try:
        world = Generic.from_robot(robot, a.world)
        arm = Arm.from_robot(robot, a.arm)
        rendering_dt = a.rendering_dt
        await world.do_command({"command": "pause"})
        home = await _joints(arm)
        report = {"rendering_dt": rendering_dt}
        checks = {
            "step": lambda: check_step(world, rendering_dt),
            "teleport": lambda: check_teleport(arm, world),
            "throughput": lambda: check_throughput(arm, world, a.prop, home, a.states),
            "concurrent": lambda: check_concurrent(world, a.prop, rendering_dt),
        }
        if a.gripper:
            checks["episode"] = lambda: check_episode(arm, world, a.prop, home, a)
        for name, check in checks.items():
            try:
                report[name] = await check()
            except Exception as exc:
                report[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        await _set(arm, world, a.prop, *_state(home, 0))
        await world.do_command({"command": "play"})
        print(json.dumps(report, indent=2))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
