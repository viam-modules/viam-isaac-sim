"""Replaying recorded states: world DoCommands stay off the event loop."""

import asyncio
import threading

from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimManager


def test_world_prop_commands_do_not_block_the_event_loop():
    # Each call waits for the other at a barrier: run on the event loop,
    # the first would block it and the second could never start.
    barrier = threading.Barrier(3, timeout=5.0)

    class BlockingSim:
        def get_prop_pose(self, prop):
            barrier.wait()
            return {"position": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]}

        def set_prop_pose(self, prop, position, orientation):
            barrier.wait()
            return {"position": position, "orientation_wxyz": orientation}

        def park_props(self, names):
            barrier.wait()
            return {"poses": {}}

    async def main():
        world = IsaacWorld("sim-world")
        return await asyncio.gather(
            world.do_command({"command": "get_prop_pose", "prop": "a"}),
            world.do_command(
                {
                    "command": "set_prop_pose",
                    "prop": "b",
                    "position": [1, 2, 3],
                    "orientation_wxyz": [1, 0, 0, 0],
                }
            ),
            world.do_command({"command": "park_props", "props": ["c"]}),
        )

    previous = SimManager._instance
    SimManager._instance = BlockingSim()
    try:
        got, placed, parked = asyncio.run(main())
    finally:
        SimManager._instance = previous
    assert got["prop"] == "a"
    assert placed["position"] == [1, 2, 3]
    assert parked == {"poses": {}}
