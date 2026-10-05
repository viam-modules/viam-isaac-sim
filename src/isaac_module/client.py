"""Client for the viam_isaac_server Kit extension.

Thread-safe: component models call in from worker threads
(asyncio.to_thread), many requests can be in flight on the one connection,
and a reader thread matches responses back to their callers. A dropped
connection fails whatever was in flight; the next call reconnects.
"""

import itertools
import socket
import threading
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable, Dict, Optional, Tuple

from viam.logging import getLogger

from . import protocol

LOGGER = getLogger("viam-isaac-sim.client")


class SimError(RuntimeError):
    """An error reported by the extension."""


class NotAttachedError(SimError):
    """The extension doesn't know the component, e.g. Isaac Sim restarted."""


class NotReadyError(SimError):
    """The simulation can't serve the request right now, e.g. it's stopped."""


def _raise_remote(method: str, error: Dict[str, Any]) -> None:
    kind = error.get("type")
    message = error.get("message") or str(kind)
    if kind == protocol.ERR_NOT_ATTACHED:
        raise NotAttachedError(message)
    if kind == protocol.ERR_NOT_READY:
        raise NotReadyError(message)
    if kind == protocol.ERR_INVALID_ARGUMENT:
        raise ValueError(message)
    if kind == protocol.ERR_UNIMPLEMENTED:
        raise NotImplementedError(message)
    raise SimError(f"isaac sim: {method} failed: {message}")


class SimClient:
    def __init__(
        self,
        address: str,
        on_connect: Optional[Callable[["SimClient"], None]] = None,
        connect_timeout: float = 10.0,
    ) -> None:
        self.address = address
        self._host, self._port = protocol.parse_address(address)
        self.connect_timeout = connect_timeout
        self._on_connect = on_connect
        self._ids = itertools.count(1)
        # held across a whole connect + handshake so only one happens at once
        self._connect_lock = threading.Lock()
        # guards _sock/_pending/_closed; never held while waiting
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._pending: Dict[int, Tuple[socket.socket, Future]] = {}
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        self._ensure_connected()

    def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Any:
        return self.call_with_payload(method, params, timeout)[0]

    def call_with_payload(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Tuple[Any, bytes]:
        sock = self._ensure_connected()
        return self._request(sock, method, params or {}, timeout)

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            sock = self._sock
        if sock is not None:
            self._drop(sock, ConnectionError("client closed"))

    def _ensure_connected(self) -> socket.socket:
        sock = self._sock
        if sock is not None:
            return sock
        with self._connect_lock:
            if self._closed:
                raise ConnectionError("client closed")
            if self._sock is not None:
                return self._sock
            try:
                sock = socket.create_connection(
                    (self._host, self._port), timeout=self.connect_timeout
                )
            except OSError as e:
                raise ConnectionError(
                    f"can't reach the Isaac Sim extension at {self.address}: {e}. "
                    "Is Isaac Sim running with the viam_isaac_server extension "
                    "enabled?"
                ) from None
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._state_lock:
                self._sock = sock
            threading.Thread(
                target=self._read_loop,
                args=(sock,),
                name="isaac-sim-client",
                daemon=True,
            ).start()
            try:
                info, _ = self._request(
                    sock,
                    "hello",
                    {"protocol_version": protocol.PROTOCOL_VERSION},
                    self.connect_timeout,
                )
            except BaseException:
                self._drop(sock, ConnectionError("handshake failed"))
                raise
            LOGGER.info(
                "connected to viam_isaac_server %s at %s",
                info.get("version"),
                self.address,
            )
        if self._on_connect is not None:
            try:
                self._on_connect(self)
            except Exception:
                LOGGER.exception("error setting up the isaac sim connection")
        return sock

    def _request(
        self, sock: socket.socket, method: str, params: Dict[str, Any], timeout: float
    ) -> Tuple[Any, bytes]:
        req_id = next(self._ids)
        fut: Future = Future()
        with self._state_lock:
            if self._sock is not sock:
                raise ConnectionError(f"connection to isaac sim at {self.address} lost")
            self._pending[req_id] = (sock, fut)
        try:
            frame = protocol.encode_frame(
                {"id": req_id, "method": method, "params": params}
            )
            try:
                with self._send_lock:
                    sock.sendall(frame)
            except OSError as e:
                self._drop(
                    sock,
                    ConnectionError(
                        f"connection to isaac sim at {self.address} lost: {e}"
                    ),
                )
            try:
                message, payload = fut.result(timeout=timeout)
            except FutureTimeout:
                raise TimeoutError(
                    f"isaac sim did not answer {method} within {timeout}s"
                ) from None
        finally:
            with self._state_lock:
                self._pending.pop(req_id, None)
        if "error" in message:
            _raise_remote(method, message["error"])
        return message.get("result"), payload

    def _read_loop(self, sock: socket.socket) -> None:
        error: Exception = ConnectionError(
            f"isaac sim at {self.address} closed the connection"
        )
        try:
            while True:
                frame = protocol.read_frame(sock)
                if frame is None:
                    break
                with self._state_lock:
                    entry = self._pending.get(frame[0].get("id"))
                if entry is not None:
                    try:
                        entry[1].set_result(frame)
                    except InvalidStateError:
                        pass
        except (OSError, protocol.ProtocolError) as e:
            error = ConnectionError(
                f"connection to isaac sim at {self.address} lost: {e}"
            )
        self._drop(sock, error)

    def _drop(self, sock: socket.socket, error: Exception) -> None:
        with self._state_lock:
            was_current = self._sock is sock
            if was_current:
                self._sock = None
            failed = [fut for s, fut in self._pending.values() if s is sock]
            closed = self._closed
        for fut in failed:
            try:
                fut.set_exception(error)
            except InvalidStateError:
                pass
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
        if was_current and not closed:
            LOGGER.warning("%s", error)
