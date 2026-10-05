"""The extension's network server, kept free of Isaac imports so it can be
tested outside of Kit.

Connections are read and written on background threads, but every request
is handled on the asyncio loop the server was started with. Inside Isaac
that is Kit's main-thread loop - the only thread Isaac's APIs may be called
from - so backend handlers can use Isaac directly, and can `await` app
updates when they need the simulation to advance.

A backend is any object with methods named rpc_<kind>_<op>(ctx, **params),
served as "<kind>.<op>"; ctx is the calling Connection. It may also define
on_disconnect(ctx).
"""

import asyncio
import inspect
import itertools
import logging
import queue
import socket
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from . import protocol

LOGGER = logging.getLogger("viam_isaac_server")

SERVER_VERSION = "0.1.0"


class NotAttached(Exception):
    """The named component isn't attached (or its prim went away)."""


class NotReady(Exception):
    """The simulation can't serve the request right now, e.g. it's stopped."""


class BadRequest(ValueError):
    pass


@dataclass
class Binary:
    """A handler result carrying a binary payload. When `encode` is set it
    runs on the connection's writer thread - keeping work like image
    compression off the simulation thread - and returns (result, payload)."""

    result: Any = None
    payload: bytes = b""
    encode: Optional[Callable[[], Tuple[Any, bytes]]] = None


def _error(e: BaseException) -> Dict[str, str]:
    if isinstance(e, NotAttached):
        kind = protocol.ERR_NOT_ATTACHED
    elif isinstance(e, NotReady):
        kind = protocol.ERR_NOT_READY
    elif isinstance(e, NotImplementedError):
        kind = protocol.ERR_UNIMPLEMENTED
    elif isinstance(e, ValueError):
        kind = protocol.ERR_INVALID_ARGUMENT
    else:
        kind = protocol.ERR_INTERNAL
    return {"type": kind, "message": str(e) or type(e).__name__}


def _collect_handlers(backend: Any) -> Dict[str, Callable[..., Any]]:
    handlers = {}
    for attr in dir(backend):
        if attr.startswith("rpc_"):
            kind, _, op = attr[len("rpc_") :].partition("_")
            handlers[f"{kind}.{op}"] = getattr(backend, attr)
    return handlers


class Connection:
    """One module connection: a reader thread hands requests to the loop, a
    writer thread sends the responses back as they complete."""

    _ids = itertools.count(1)

    def __init__(self, server: "Server", sock: socket.socket, peer: Any) -> None:
        self.id = next(Connection._ids)
        self.peer = peer
        self._server = server
        self._sock = sock
        self._out: "queue.Queue[Optional[Tuple[Any, Future]]]" = queue.Queue()
        self._closed = threading.Event()

    def start(self) -> None:
        for target, role in ((self._read_loop, "read"), (self._write_loop, "write")):
            threading.Thread(
                target=target, name=f"viam-isaac-{role}-{self.id}", daemon=True
            ).start()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._out.put(None)
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def _read_loop(self) -> None:
        try:
            while True:
                frame = protocol.read_frame(self._sock)
                if frame is None:
                    break
                self._submit(frame[0])
        except (OSError, protocol.ProtocolError) as e:
            if not self._closed.is_set():
                LOGGER.warning("dropping connection from %s: %s", self.peer, e)
        finally:
            self.close()
            self._server._on_closed(self)

    def _submit(self, message: Dict[str, Any]) -> None:
        req_id = message.get("id")
        coro = self._server._dispatch(
            self, message.get("method"), message.get("params")
        )
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, self._server.loop)
        except RuntimeError as e:  # the loop is gone; we're shutting down
            coro.close()
            fut = Future()
            fut.set_exception(e)
        fut.add_done_callback(lambda f: self._out.put((req_id, f)))

    def _write_loop(self) -> None:
        while True:
            item = self._out.get()
            if item is None:
                return
            req_id, fut = item
            try:
                self._sock.sendall(self._response(req_id, fut))
            except OSError:
                self.close()
                return

    @staticmethod
    def _response(req_id: Any, fut: Future) -> bytes:
        try:
            result = fut.result()
            payload = b""
            if isinstance(result, Binary):
                if result.encode is not None:
                    result, payload = result.encode()
                else:
                    result, payload = result.result, result.payload
            return protocol.encode_frame({"id": req_id, "result": result}, payload)
        except BaseException as e:
            return protocol.encode_frame({"id": req_id, "error": _error(e)})


class Server:
    def __init__(
        self,
        backend: Any,
        loop: asyncio.AbstractEventLoop,
        host: str = "127.0.0.1",
        port: int = protocol.DEFAULT_PORT,
    ) -> None:
        self.backend = backend
        self.loop = loop
        self.host = host
        self.port = port
        self._handlers = _collect_handlers(backend)
        self._listener: Optional[socket.socket] = None
        self._stopping = threading.Event()
        self._conns: Dict[int, Connection] = {}
        self._lock = threading.Lock()

    @property
    def address(self) -> Tuple[str, int]:
        """(host, port) actually bound - port 0 picks a free one."""
        assert self._listener is not None, "server not started"
        host, port = self._listener.getsockname()[:2]
        return host, port

    def start(self) -> None:
        self._stopping.clear()
        listener = socket.create_server((self.host, self.port))
        # accept() isn't reliably interrupted by close() from another thread,
        # so poll for stop() instead
        listener.settimeout(0.5)
        self._listener = listener
        threading.Thread(
            target=self._accept_loop,
            args=(listener,),
            name="viam-isaac-accept",
            daemon=True,
        ).start()
        LOGGER.info("viam_isaac_server listening on %s:%d", *self.address)

    def stop(self) -> None:
        self._stopping.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.close()
        with self._lock:
            conns = list(self._conns.values())
        for conn in conns:
            conn.close()

    def _accept_loop(self, listener: socket.socket) -> None:
        while not self._stopping.is_set():
            try:
                sock, peer = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self._stopping.is_set():
                    LOGGER.exception("viam_isaac_server stopped accepting connections")
                return
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn = Connection(self, sock, peer)
            with self._lock:
                self._conns[conn.id] = conn
            LOGGER.info("viam module connected from %s", peer)
            conn.start()

    def _on_closed(self, conn: Connection) -> None:
        with self._lock:
            self._conns.pop(conn.id, None)
        LOGGER.info("viam module at %s disconnected", conn.peer)
        hook = getattr(self.backend, "on_disconnect", None)
        if hook is None:
            return

        def _run() -> None:
            try:
                hook(conn)
            except Exception:
                LOGGER.exception("error cleaning up after %s", conn.peer)

        try:
            self.loop.call_soon_threadsafe(_run)
        except RuntimeError:
            pass  # loop closed; nothing left to clean up

    async def _dispatch(self, conn: Connection, method: Any, params: Any) -> Any:
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise BadRequest(f"{method}: params must be an object")
        if method == "hello":
            return self._hello(**params)
        fn = self._handlers.get(method) if isinstance(method, str) else None
        if fn is None:
            raise BadRequest(f"unknown method {method!r}")
        try:
            bound = inspect.signature(fn).bind(conn, **params)
        except TypeError as e:
            raise BadRequest(f"{method}: {e}") from None
        try:
            result = fn(*bound.args, **bound.kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except (NotAttached, NotReady, NotImplementedError, ValueError):
            raise
        except Exception:
            LOGGER.exception("%s failed", method)
            raise

    @staticmethod
    def _hello(protocol_version: Any = None, **_: Any) -> Dict[str, Any]:
        if protocol_version != protocol.PROTOCOL_VERSION:
            raise BadRequest(
                f"protocol mismatch: the module speaks version {protocol_version}, "
                f"this extension speaks {protocol.PROTOCOL_VERSION}; update the "
                "older of the two"
            )
        return {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "version": SERVER_VERSION,
        }
