# tests/test_pointcloud.py
import struct

import numpy as np
import pytest

from isaac_module.pointcloud import (
    encode_depth,
    optical_to_axes,
    to_pcd,
    unproject,
)


def parse_pcd(data: bytes):
    """Minimal binary PCD reader for tests: (header dict, structured array)."""
    header_end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    header = {}
    for line in data[:header_end].decode("ascii").splitlines():
        key, _, value = line.partition(" ")
        header[key] = value
    fields = header["FIELDS"].split()
    types = {"F": "<f4", "I": "<u4"}
    dtype = [(f, types[t]) for f, t in zip(fields, header["TYPE"].split())]
    return header, np.frombuffer(data[header_end:], dtype=dtype)


def test_unproject_known_pixel_and_principal_point():
    depth = np.full((4, 6), 2.0, dtype=np.float32)
    intr = (100.0, 100.0, 3.0, 2.0)  # fx, fy, cx, cy
    points, colors = unproject(depth, intr)
    assert colors is None
    assert points.shape == (24, 3)
    # row-major: pixel (v=2, u=3) is the principal point
    assert points[2 * 6 + 3] == pytest.approx((0.0, 0.0, 2.0))
    # pixel (v=0, u=5): x=(5-3)*2/100, y=(0-2)*2/100
    assert points[5] == pytest.approx((0.04, -0.04, 2.0))


def test_unproject_drops_invalid_and_far_and_keeps_colors():
    depth = np.array([[1.0, np.nan, np.inf, 0.0, -1.0, 5.0]], dtype=np.float32)
    rgb = np.arange(18, dtype=np.uint8).reshape(1, 6, 3)
    points, colors = unproject(depth, (1.0, 1.0, 0.0, 0.0), rgb, max_depth_m=3.0)
    assert len(points) == 1
    assert points[0][2] == pytest.approx(1.0)
    assert colors.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize(
    "axes, expected",
    [
        ("ros", (1.0, 2.0, 3.0)),
        ("world", (3.0, -1.0, -2.0)),
        ("usd", (1.0, -2.0, -3.0)),
    ],
)
def test_optical_to_axes(axes, expected):
    out = optical_to_axes(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), axes)
    assert out[0] == pytest.approx(expected)


def test_to_pcd_uncolored_round_trip():
    pts = np.array([[1.0, 2.0, 3.0], [-1.5, 0.25, 4.0]], dtype=np.float32)
    header, arr = parse_pcd(to_pcd(pts))
    assert header["FIELDS"] == "x y z"
    assert header["POINTS"] == "2"
    assert np.stack([arr["x"], arr["y"], arr["z"]], axis=1) == pytest.approx(pts)


def test_to_pcd_colored_round_trip():
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    header, arr = parse_pcd(to_pcd(pts, np.array([[255, 128, 1]], dtype=np.uint8)))
    assert header["FIELDS"] == "x y z rgb"
    assert int(arr["rgb"][0]) == (255 << 16) | (128 << 8) | 1


def test_to_pcd_zero_points():
    header, arr = parse_pcd(to_pcd(np.zeros((0, 3), dtype=np.float32)))
    assert header["POINTS"] == "0"
    assert len(arr) == 0


def test_encode_depth_header_and_values():
    depth = np.array([[1.0, np.nan], [70.0, 0.5]], dtype=np.float32)
    data = encode_depth(depth, max_depth_m=None)
    assert data[:8] == b"DEPTHMAP"
    assert struct.unpack(">qq", data[8:24]) == (2, 2)  # width, height
    mm = np.frombuffer(data[24:], dtype=">u2").reshape(2, 2)
    assert mm.tolist() == [[1000, 0], [0, 500]]  # nan and >65.535 m -> 0


def test_encode_depth_respects_max_depth():
    data = encode_depth(np.array([[1.0, 2.0]], dtype=np.float32), max_depth_m=1.5)
    assert np.frombuffer(data[24:], dtype=">u2").tolist() == [1000, 0]
