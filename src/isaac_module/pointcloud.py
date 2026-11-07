"""Depth-image and point-cloud helpers. numpy only - no Isaac imports, so the
mock sim and unit tests exercise exactly this code."""

import struct
from typing import Optional, Tuple

import numpy as np

Intrinsics = Tuple[float, float, float, float]  # fx, fy, cx, cy in pixels

# Optical (X right, Y down, Z forward) -> each Isaac camera_axes convention;
# rows are the target axes expressed in optical coordinates.
_OPTICAL_TO = {
    "ros": np.eye(3, dtype=np.float32),
    # world: X forward, Y left, Z up
    "world": np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32),
    # usd: X right, Y up, -Z forward
    "usd": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
}


def unproject(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    rgb: Optional[np.ndarray] = None,
    max_depth_m: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Depth (H, W) meters to (N, 3) optical-frame points (+ (N, 3) colors)."""
    fx, fy, cx, cy = intrinsics
    v, u = np.indices(depth.shape)
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    if max_depth_m is not None:
        valid &= z <= max_depth_m
    z = z[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[valid].astype(np.uint8) if rgb is not None else None
    return points, colors


def optical_to_axes(points: np.ndarray, axes: str) -> np.ndarray:
    return (points @ _OPTICAL_TO[axes].T).astype(np.float32, copy=False)


def to_pcd(points: np.ndarray, colors: Optional[np.ndarray] = None) -> bytes:
    """Binary PCD v0.7 in meters, the format viam-server reads."""
    pts = np.ascontiguousarray(points, dtype="<f4").reshape(-1, 3)
    n = len(pts)
    if colors is None:
        fields, size, types, count = "x y z", "4 4 4", "F F F", "1 1 1"
        body = pts.tobytes()
    else:
        c = np.asarray(colors, dtype=np.uint32).reshape(-1, 3)
        rec = np.empty(n, dtype=[("xyz", "<f4", 3), ("rgb", "<u4")])
        rec["xyz"] = pts
        rec["rgb"] = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        fields, size, types, count = "x y z rgb", "4 4 4 4", "F F F I", "1 1 1 1"
        body = rec.tobytes()
    header = (
        f"VERSION .7\nFIELDS {fields}\nSIZE {size}\nTYPE {types}\nCOUNT {count}\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    return header.encode("ascii") + body


def encode_depth(depth_m: np.ndarray, max_depth_m: Optional[float] = None) -> bytes:
    """Viam depth map (image/vnd.viam.dep): DEPTHMAP, int64 BE width, height,
    then big-endian uint16 millimeters. Invalid / unrepresentable -> 0."""
    d = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0) & (d * 1000.0 <= 65535)
    if max_depth_m is not None:
        valid &= d <= max_depth_m
    mm = np.where(valid, np.round(d * 1000.0), 0).astype(">u2")
    h, w = mm.shape
    return b"DEPTHMAP" + struct.pack(">qq", w, h) + mm.tobytes()
