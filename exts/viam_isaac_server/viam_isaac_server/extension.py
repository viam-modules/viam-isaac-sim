"""Kit entry point: serves the viam module for as long as the extension is
enabled."""

import asyncio
import logging
from typing import Optional

import carb.settings
import omni.ext

from .isaac_backend import IsaacBackend
from .protocol import DEFAULT_PORT
from .server import Server

LOGGER = logging.getLogger("viam_isaac_server")

_SETTINGS = "/exts/viam_isaac_server"


class ViamIsaacServerExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        settings = carb.settings.get_settings()
        host = settings.get(f"{_SETTINGS}/host") or "127.0.0.1"
        port = int(settings.get(f"{_SETTINGS}/port") or DEFAULT_PORT)

        self._backend: Optional[IsaacBackend] = IsaacBackend()
        # Kit's main-thread loop: requests are handled there, between updates
        self._server: Optional[Server] = Server(
            self._backend, asyncio.get_event_loop(), host, port
        )
        try:
            self._server.start()
        except OSError as e:
            LOGGER.error(
                "viam_isaac_server could not listen on %s:%d: %s", host, port, e
            )
            self._server = None

    def on_shutdown(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
        if self._backend is not None:
            self._backend.shutdown()
            self._backend = None
