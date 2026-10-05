"""Wire format between the viam module and the viam_isaac_server extension.

This file is shared verbatim by both sides (the module keeps a copy at
src/isaac_module/protocol.py; tests/test_remote.py checks they match), so it
must stay stdlib-only.

Every message is one frame:

    !II header       - byte lengths of the JSON part and the binary payload
    JSON (utf-8)     - request:  {"id": n, "method": "arm.attach", "params": {...}}
                       response: {"id": n, "result": ...} or
                                 {"id": n, "error": {"type": ..., "message": ...}}
    payload (bytes)  - optional, e.g. an encoded camera image

Requests carry ids so a client can have many in flight on one connection;
responses may come back in any order.
"""

import json
import socket
import struct
from typing import Any, Dict, Optional, Tuple

PROTOCOL_VERSION = 1
DEFAULT_PORT = 47800

# error "type"s; the client maps these back to python exceptions
ERR_NOT_ATTACHED = "not_attached"  # unknown component name (e.g. isaac restarted)
ERR_NOT_READY = "not_ready"  # e.g. the timeline is stopped
ERR_INVALID_ARGUMENT = "invalid_argument"
ERR_UNIMPLEMENTED = "unimplemented"
ERR_INTERNAL = "internal"

# camera frames the extension couldn't compress (no PIL in Isaac's python):
# the payload is height*width*3 bytes of RGB, and the module encodes it
MIME_RAW_RGB = "image/x-viam-raw-rgb"

_HEADER = struct.Struct("!II")
# sanity limits so a stray non-protocol client (an HTTP probe, say) is
# rejected instead of making us allocate gigabytes
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024


class ProtocolError(Exception):
    pass


def encode_frame(message: Dict[str, Any], payload: bytes = b"") -> bytes:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(len(body), len(payload)) + body + payload


def _recv_exactly(sock: socket.socket, n: int) -> Optional[bytes]:
    """n bytes from sock, or None if the peer closed before sending any."""
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            if remaining == n:
                return None
            raise ProtocolError("connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> Optional[Tuple[Dict[str, Any], bytes]]:
    """The next (message, payload) from sock, or None on a clean close."""
    header = _recv_exactly(sock, _HEADER.size)
    if header is None:
        return None
    json_len, payload_len = _HEADER.unpack(header)
    if json_len > MAX_JSON_BYTES or payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"frame too large ({json_len}+{payload_len} bytes); "
            "is the peer speaking this protocol?"
        )
    body = _recv_exactly(sock, json_len) if json_len else b""
    payload = _recv_exactly(sock, payload_len) if payload_len else b""
    if body is None or payload is None:
        raise ProtocolError("connection closed mid-frame")
    try:
        message = json.loads(body.decode("utf-8"))
    except ValueError as e:
        raise ProtocolError(f"malformed frame: {e}") from e
    if not isinstance(message, dict):
        raise ProtocolError("malformed frame: expected a JSON object")
    return message, payload


def parse_address(address: str) -> Tuple[str, int]:
    """'host', 'host:port' or ':port' -> (host, port)."""
    address = address.strip()
    if not address:
        return "localhost", DEFAULT_PORT
    host, sep, port = address.rpartition(":")
    if not sep:
        return address, DEFAULT_PORT
    try:
        port_num = int(port)
    except ValueError:
        raise ValueError(f"invalid address {address!r}: port must be a number")
    if not 0 < port_num < 65536:
        raise ValueError(f"invalid address {address!r}: port out of range")
    return host.strip("[]") or "localhost", port_num
