#!/usr/bin/env python3
"""Measure Lite6 simulator target following at the policy's 20 Hz cadence."""

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from viam.components.arm import Arm, JointPositions
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

CONTROL_HZ = 20.0
PERIOD_S = 1.0 / CONTROL_HZ
PHYSICAL_LIMITS = {
    "source": "UF Lite6 Hardware Manual V2.6.0 (2025-05), page 2",
    "url": "https://www.ufactory.cc/wp-content/uploads/2025/05/Lite6_Hardware_Manual_V2.6.0.pdf",
    "joint_speed_rad_s": math.pi,
    "joint_acceleration_rad_s2": math.radians(1145.0),
    "applies_to_simulator": False,
}
POSES_DEG = (
    (0.0, 0.0, 90.0, 0.0, 90.0, 0.0),
    (10.0, 10.0, 80.0, 10.0, 80.0, 10.0),
    (-10.0, -10.0, 100.0, -10.0, 100.0, -10.0),
)


async def _joint_radians(arm: Arm) -> np.ndarray:
    response = await arm.do_command({"command": "get_joint_positions_radians"})
    values = np.asarray(response.get("values"), dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise RuntimeError("arm returned invalid six-joint measurement")
    return values


async def _sample_transition(
    arm: Arm, start: np.ndarray, target: np.ndarray, duration_s: float
) -> Mapping[str, Any]:
    await arm.do_command(
        {"command": "set_joint_targets_radians", "values": target.tolist()}
    )
    started = time.monotonic()
    deadline = started
    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    while time.monotonic() - started <= duration_s:
        delay = deadline - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        timestamps.append(time.monotonic() - started)
        positions.append(await _joint_radians(arm))
        deadline += PERIOD_S
    samples = np.asarray(positions)
    elapsed = np.asarray(timestamps)
    dt = np.diff(elapsed)
    velocity = np.diff(samples, axis=0) / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / dt[1:, None]
    low = np.minimum(start, target)
    high = np.maximum(start, target)
    overshoot = np.maximum(np.maximum(low - samples, samples - high), 0.0)
    tracking_error = np.abs(samples - target)

    await arm.stop()
    before_stop = await _joint_radians(arm)
    await asyncio.sleep(0.5)
    after_stop = await _joint_radians(arm)
    return {
        "start_rad": start.tolist(),
        "target_rad": target.tolist(),
        "sample_count": int(samples.shape[0]),
        "sample_times_s": elapsed.tolist(),
        "positions_rad": samples.tolist(),
        "max_abs_velocity_rad_s": np.max(np.abs(velocity), axis=0).tolist(),
        "max_abs_acceleration_rad_s2": np.max(
            np.abs(acceleration), axis=0
        ).tolist(),
        "max_tracking_error_rad": np.max(tracking_error, axis=0).tolist(),
        "final_tracking_error_rad": tracking_error[-1].tolist(),
        "overshoot_rad": np.max(overshoot, axis=0).tolist(),
        "stop_displacement_rad": np.abs(after_stop - before_stop).tolist(),
    }


async def run(args: argparse.Namespace) -> Mapping[str, Any]:
    api_key = os.environ.get("VIAM_API_KEY")
    api_key_id = os.environ.get("VIAM_API_KEY_ID")
    if bool(api_key) != bool(api_key_id):
        raise ValueError("set both VIAM_API_KEY and VIAM_API_KEY_ID, or neither")
    options = (
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
        if api_key and api_key_id
        else RobotClient.Options(
            dial_options=DialOptions(insecure=not args.secure)
        )
    )
    if args.disable_webrtc:
        assert options.dial_options is not None
        options.dial_options.disable_webrtc = True

    async with await RobotClient.at_address(args.address, options) as machine:
        arm = Arm.from_robot(machine, args.arm)
        metadata = await arm.do_command({"command": "dynamics_metadata"})
        transitions = []
        await arm.move_to_joint_positions(JointPositions(values=list(POSES_DEG[0])))
        for pose_deg in (*POSES_DEG[1:], POSES_DEG[0]):
            start = await _joint_radians(arm)
            target = np.radians(np.asarray(pose_deg, dtype=np.float64))
            transitions.append(
                await _sample_transition(arm, start, target, args.duration_sec)
            )
        result = {
            "schema_version": 1,
            "control_hz": CONTROL_HZ,
            "simulator_controller": metadata,
            "physical_limits_reference": PHYSICAL_LIMITS,
            "transitions": transitions,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="machine address from the Viam CONNECT tab")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument(
        "--output", default="runs/live/lite6-dynamics-measurement.json"
    )
    parser.add_argument(
        "--secure", action="store_true", help="use TLS for a direct machine address"
    )
    parser.add_argument("--disable-webrtc", action="store_true")
    args = parser.parse_args()
    if args.duration_sec < 0.5:
        parser.error("--duration-sec must be at least 0.5")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
