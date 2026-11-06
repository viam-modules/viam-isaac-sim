"""
Map the office-coffee-sim scene with nav-stack SLAM and the Jetbot's lidar.

Drives the Jetbot in a loop around both tables, steering from base-odom's sim
ground truth, while the `slam` service (viam-labs:nav-stack:slam, mapping mode)
builds the map from base-lidar. At each waypoint it prints the SLAM pose next
to ground truth, then saves the map and renders its occupied cells over the
scene's real walls and tables.

    VIAM_API_KEY=... VIAM_API_KEY_ID=... python3 examples/slam_mapping.py <machine-address>
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from viam.components.base import Base
from viam.components.movement_sensor import MovementSensor
from viam.robot.client import RobotClient
from viam.services.slam import SLAMClient

# A loop around both tables (world meters), clear of walls and the bookshelf.
WAYPOINTS = [(0.6, -0.9), (3.5, -0.9), (3.5, 1.0), (-1.0, 1.0), (-1.0, -0.9), (0.6, -0.9)]
DRIVE_MM_PER_SEC = 150.0
SPIN_DEG_PER_SEC = 30.0
SEGMENT_M = 0.4
ARRIVE_M = 0.15
MAX_LEGS = 40
# The slam service maps with map_when_still: it only takes a scan once the base
# has been stopped for its dwell (1 s), so hold still after every move.
SETTLE_SEC = 1.5
# Footprints of the scene's fixed props (center x, y, size x, y), for the plot.
SCENE_BOXES = {
    "pick_table": (0.0, 0.0, 0.8, 0.6),
    "drop_table": (2.5, 0.0, 0.8, 0.6),
    "bookshelf": (0.2, 1.8, 1.2, 0.35),
    "wall_north": (1.25, 2.05, 5.6, 0.1),
    "wall_south": (1.25, -2.05, 5.6, 0.1),
    "wall_west": (-1.55, 0.0, 0.1, 4.2),
    "wall_east": (4.05, 0.0, 0.1, 4.2),
}


async def truth(odom: MovementSensor) -> tuple[float, float, float]:
    r = await odom.get_readings()
    return r["sim_position_m"]["x"], r["sim_position_m"]["y"], float(r["yaw_deg"])


async def go_to(base: Base, odom: MovementSensor, x: float, y: float) -> None:
    """Spin and move_straight are open loop and the Jetbot under/overshoots
    both, so re-aim from ground truth every SEGMENT_M until close enough."""
    for _ in range(MAX_LEGS):
        px, py, yaw = await truth(odom)
        dist = math.hypot(x - px, y - py)
        if dist < ARRIVE_M:
            return
        turn = (math.degrees(math.atan2(y - py, x - px)) - yaw + 180.0) % 360.0 - 180.0
        if abs(turn) > 3.0:
            await base.spin(turn, SPIN_DEG_PER_SEC)
            await asyncio.sleep(SETTLE_SEC)
            continue  # re-check heading before driving
        await base.move_straight(int(min(dist, SEGMENT_M) * 1000.0), DRIVE_MM_PER_SEC)
        await asyncio.sleep(SETTLE_SEC)
    raise RuntimeError(f"could not reach ({x:.2f}, {y:.2f}); stuck at {await truth(odom)}")


def read_pcd_xy(data: bytes | list[bytes]) -> np.ndarray:
    if isinstance(data, list):  # the SLAM API streams the map in chunks
        data = b"".join(data)
    end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    header = dict(line.split(" ", 1) for line in data[:end].decode().splitlines())
    fields = header["FIELDS"].split()
    sizes = [int(s) for s in header["SIZE"].split()]
    types = {"F": "f", "I": "i", "U": "u"}
    dtype = [
        (f, f"<{types[t]}{s}")
        for f, t, s in zip(fields, header["TYPE"].split(), sizes)
    ]
    arr = np.frombuffer(data[end:], dtype=dtype)
    return np.stack([arr["x"], arr["y"]], axis=1).astype(np.float64)


def plot(points_m: np.ndarray, start: tuple[float, float, float], path: list, out: str) -> None:
    """Map frame starts at the robot's start pose; shift it into the world."""
    sx, sy, syaw = start
    c, s = math.cos(math.radians(syaw)), math.sin(math.radians(syaw))
    world = np.stack(
        [sx + c * points_m[:, 0] - s * points_m[:, 1], sy + s * points_m[:, 0] + c * points_m[:, 1]],
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    for (cx, cy, w, h) in SCENE_BOXES.values():
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="tab:blue", lw=1))
    ax.scatter(world[:, 0], world[:, 1], s=2, c="k", label="SLAM occupied cells")
    tx, ty = zip(*path)
    ax.plot(tx, ty, "-", c="tab:orange", lw=1, label="driven path (ground truth)")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 4.5)
    ax.set_ylim(-2.5, 2.5)
    ax.legend(loc="upper right")
    ax.set_title("nav-stack SLAM map vs. scene (blue outlines)")
    fig.savefig(out, dpi=110, bbox_inches="tight")


async def run(machine: RobotClient, out: str, map_name: str) -> None:
    base = Base.from_robot(machine, "base")
    odom = MovementSensor.from_robot(machine, "base-odom")
    slam = SLAMClient.from_robot(machine, "slam")

    await base.stop()
    print("start_mapping:", await slam.do_command({"command": "start_mapping", "map": map_name}))
    start = await truth(odom)
    path = [start[:2]]
    for wx, wy in WAYPOINTS:
        await go_to(base, odom, wx, wy)
        gt = await truth(odom)
        est = await slam.get_position()
        path.append(gt[:2])
        # the map frame starts at the start pose; compare there
        c, s = math.cos(math.radians(-start[2])), math.sin(math.radians(-start[2]))
        dx, dy = gt[0] - start[0], gt[1] - start[1]
        mx, my = c * dx - s * dy, s * dx + c * dy
        err = math.hypot(est.x / 1000 - mx, est.y / 1000 - my)
        print(
            f"waypoint ({wx:.1f}, {wy:.1f}): truth in map ({mx:.2f}, {my:.2f}, "
            f"{gt[2] - start[2]:.0f} deg)  slam ({est.x / 1000:.2f}, {est.y / 1000:.2f}, "
            f"{est.theta:.0f} deg)  error {err:.2f} m"
        )

    print("save_map:", await slam.do_command({"command": "save_map"}))
    pcd = await slam.get_point_cloud_map()
    points = read_pcd_xy(pcd)
    print(f"map has {len(points)} occupied cells")
    plot(points, start, path, out)
    print(f"wrote {out}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("address")
    parser.add_argument("--out", default="slam_map.png")
    parser.add_argument("--map", default="office", help="nav-stack map to (re)start")
    args = parser.parse_args()
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"]
    )
    machine = await RobotClient.at_address(args.address, opts)
    try:
        await run(machine, args.out, args.map)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
