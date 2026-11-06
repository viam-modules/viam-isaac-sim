"""
Export the nav-stack SLAM service's active map as a map directory that
nav-stack can load (map.yaml + map.pgm), ready to package for the registry.

    VIAM_API_KEY=... VIAM_API_KEY_ID=... \\
        python3 examples/export_slam_map.py <machine-address> office6 --out maps

writes maps/office6/{map.yaml,map.pgm}. The keyframe pose graph isn't
exported (nav-stack has no API for it); it's only needed to extend the map,
not to localize on it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
from pathlib import Path

import numpy as np
from viam.robot.client import RobotClient
from viam.services.slam import SLAMClient


def write_map(map_dir: Path, grid: dict) -> None:
    """Same encoding as nav-stack's slam_builtin.persistence.save_occupancy:
    unknown 205, free 254, occupied 0, intermediate costs greyscale, rows
    flipped to top-down."""
    cells = np.frombuffer(base64.b64decode(grid["data"]), dtype=np.int8).astype(np.int16)
    cells = cells.reshape(int(grid["rows"]), int(grid["cols"]))
    top_down = np.flipud(cells)
    pixels = np.full(top_down.shape, 205, dtype=np.uint8)
    pixels[top_down == 0] = 254
    pixels[top_down >= 50] = 0
    mid = (top_down > 0) & (top_down < 50)
    pixels[mid] = (254 - (top_down[mid] / 100.0) * 254).astype(np.uint8)

    map_dir.mkdir(parents=True, exist_ok=True)
    h, w = pixels.shape
    (map_dir / "map.pgm").write_bytes(f"P5\n{w} {h}\n255\n".encode("ascii") + pixels.tobytes())
    (map_dir / "map.yaml").write_text(
        "image: map.pgm\n"
        "mode: trinary\n"
        f"resolution: {float(grid['cellSize']):.6f}\n"
        f"origin: [{float(grid['xMin']):.6f}, {float(grid['yMin']):.6f}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("address")
    parser.add_argument("map", help="map name to export (it must be the active map)")
    parser.add_argument("--out", default="maps", help="directory to write <map>/ into")
    parser.add_argument("--slam", default="slam")
    args = parser.parse_args()
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"]
    )
    machine = await RobotClient.at_address(args.address, opts)
    try:
        slam = SLAMClient.from_robot(machine, args.slam)
        active = (await slam.do_command({"command": "get_active_map"})).get("map")
        if active != args.map:
            raise SystemExit(f"active map is {active!r}, not {args.map!r}")
        grid = await slam.do_command({"command": "get_grid"})
        if not grid.get("data"):
            raise SystemExit(f"no grid available: {grid}")
        map_dir = Path(args.out) / args.map
        write_map(map_dir, grid)
        print(f"wrote {map_dir} ({grid['cols']}x{grid['rows']} cells at {grid['cellSize']} m)")
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
