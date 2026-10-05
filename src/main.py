"""Module entrypoint.

Isaac Sim runs in its own process with the viam_isaac_server extension
enabled (see exts/), so this is a plain viam module: component calls become
requests to that extension.
"""

import asyncio

from viam.module.module import Module

import isaac_module.models  # noqa: F401 - registers all models
from isaac_module import sdk_patches
from isaac_module.sim_manager import SimManager

sdk_patches.apply()


def main() -> None:
    try:
        asyncio.run(Module.run_from_registry())
    finally:
        SimManager.get().close()


if __name__ == "__main__":
    main()
