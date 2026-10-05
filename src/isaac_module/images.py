from io import BytesIO
from typing import Tuple

JPEG = "image/jpeg"
PNG = "image/png"


def encode_rgb(rgb, mime_type: str = "") -> Tuple[bytes, str]:
    """Encode an (H, W, 3) uint8 array: PNG if asked for, otherwise JPEG.
    Returns (data, mime type)."""
    from PIL import Image

    img = Image.fromarray(rgb)
    buf = BytesIO()
    if mime_type == PNG:
        img.save(buf, format="PNG")
        return buf.getvalue(), PNG
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), JPEG
