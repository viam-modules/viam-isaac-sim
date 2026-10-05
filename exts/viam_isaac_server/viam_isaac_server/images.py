"""Camera frame encoding. Runs on a connection's writer thread (see
server.Binary), not the simulation thread."""

from io import BytesIO
from typing import Any, Dict, Tuple

from .protocol import MIME_RAW_RGB


def encode_rgb(rgb: Any, mime_type: str) -> Tuple[Dict[str, Any], bytes]:
    """(result, payload) for an (H, W, 3) uint8 frame: PNG if asked for,
    otherwise JPEG - or raw RGB for the module to encode when this python
    has no PIL."""
    height, width = rgb.shape[:2]
    try:
        from PIL import Image
    except ImportError:
        return (
            {"mime_type": MIME_RAW_RGB, "width": width, "height": height},
            rgb.tobytes(),
        )

    img = Image.fromarray(rgb)
    buf = BytesIO()
    if mime_type == "image/png":
        img.save(buf, format="PNG")
    else:
        mime_type = "image/jpeg"
        img.save(buf, format="JPEG", quality=90)
    return {"mime_type": mime_type, "width": width, "height": height}, buf.getvalue()
