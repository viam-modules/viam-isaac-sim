"""Module entrypoint.

Isaac Sim (Omniverse Kit) must own the thread it runs on, so the layout is:
  - main thread: SimManager.main_loop() - boots and steps the simulator
  - side thread: the Viam module gRPC server

When viam-server shuts the module down, the module thread exits and asks the
sim loop to stop, which closes the SimulationApp.
"""

import asyncio
import signal
import sys
import threading

from viam.logging import getLogger
from viam.module.module import Module

import isaac_module.models
from isaac_module import sdk_patches
from isaac_module.sim_manager import SimManager

sdk_patches.apply()

LOGGER = getLogger("viam-isaac-sim.main")


def _run_module(sim: SimManager) -> None:
    loop = asyncio.new_event_loop()
    # grpclib wants to install signal handlers, which is only possible on the
    # main thread - and the main thread belongs to isaac sim. Shutdown is
    # handled by the signal handlers installed in main() instead.
    loop.add_signal_handler = lambda *args, **kwargs: None
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(Module.run_from_registry())
    except Exception:
        LOGGER.exception("module server exited with error")
    finally:
        sim.request_stop()


def main() -> None:
    sim = SimManager.get()

    def _shutdown(signum, frame):
        sim.request_stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    t = threading.Thread(target=_run_module, args=(sim,), name="viam-module", daemon=True)
    t.start()
    sim.main_loop()
    sys.exit(0)


if __name__ == "__main__":
    main()
