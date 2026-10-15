import itertools
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from isaac_module.cosmos_capture import (
    COSMOS_MODALITIES,
    finalize_cosmos_videos,
    validate_cosmos_videos,
)

ROLE_DIMENSIONS = {"fixed": (16, 12), "wrist": (12, 8)}


def _idr_frame_indices(path: Path) -> set[int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame_by_pts = {
            frame.pts: index for index, frame in enumerate(container.decode(stream))
        }
        extradata = stream.codec_context.extradata
        nal_length_size = (extradata[4] & 0x03) + 1

    result: set[int] = set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.pts not in frame_by_pts:
                continue
            payload = bytes(packet)
            offset = 0
            while offset < len(payload):
                nal_size = int.from_bytes(
                    payload[offset : offset + nal_length_size], "big"
                )
                offset += nal_length_size
                if payload[offset] & 0x1F == 5:
                    result.add(frame_by_pts[packet.pts])
                offset += nal_size
    return result


def _write_png_sequences(root: Path, frame_count: int = 3) -> None:
    for role, (width, height) in ROLE_DIMENSIONS.items():
        for modality_index, modality in enumerate(COSMOS_MODALITIES):
            image_dir = root / role / "clip_0000" / modality
            image_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in range(frame_count):
                value = 20 + frame_index * 70
                channels = 4 if modality_index % 2 else 3
                pixels = np.full((height, width, channels), value, dtype=np.uint8)
                if channels == 4:
                    pixels[:, :, 3] = 127
                Image.fromarray(pixels, mode="RGBA" if channels == 4 else "RGB").save(
                    image_dir / f"{modality}_{frame_index:04d}.png"
                )


def _write_h264_video(path: Path, *, frame_count: int, fps: int) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"
        for frame_index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(
                np.full((12, 16, 3), 20 + frame_index * 70, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _decoded_timestamps_and_markers(
    video_path: Path,
) -> tuple[list[float], list[float]]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        timestamps = []
        markers = []
        for frame in container.decode(stream):
            timestamps.append(float(frame.pts * frame.time_base))
            markers.append(float(frame.to_ndarray(format="rgb24")[0, 0, 0]))
    return timestamps, markers


def test_finalizer_encodes_all_png_modalities_at_configured_rate(tmp_path):
    _write_png_sequences(tmp_path)

    facts = finalize_cosmos_videos(
        tmp_path,
        expected_frames=3,
        sample_rate_hz=15.0,
        stage_start_frames=(0, 2),
    )

    assert json.loads(json.dumps(facts)) == facts
    assert facts["path"] == str(tmp_path.resolve())
    for role, dimensions in ROLE_DIMENSIONS.items():
        streams = facts["roles"][role]
        assert set(streams) == set(COSMOS_MODALITIES)
        for modality, stream_facts in streams.items():
            video_path = tmp_path / stream_facts["relative_path"]
            assert video_path == tmp_path / role / "clip_0000" / f"{modality}.mp4"
            assert stream_facts == {
                "relative_path": f"{role}/clip_0000/{modality}.mp4",
                "frame_count": 3,
                "width": dimensions[0],
                "height": dimensions[1],
                "codec": "h264",
                "pixel_format": "yuv420p",
                "average_fps": 15.0,
                "raw_average_fps": 15.0,
                "timestamp_spacing_s": 1.0 / 15.0,
                "duration_s": 0.2,
            }
            with av.open(str(video_path)) as container:
                assert len(container.streams.video) == 1
                stream = container.streams.video[0]
                assert stream.codec_context.name == "h264"
                assert stream.codec_context.format.name == "yuv420p"
                assert stream.average_rate == Fraction(15, 1)
                assert {0, 2} <= _idr_frame_indices(video_path)
            timestamps, markers = _decoded_timestamps_and_markers(video_path)
            assert timestamps == pytest.approx([0.0, 1 / 15, 2 / 15], abs=1e-9)
            assert [
                later - earlier for earlier, later in itertools.pairwise(timestamps)
            ] == pytest.approx([1 / 15, 1 / 15], abs=1e-9)
            assert markers == pytest.approx([20.0, 90.0, 160.0], abs=3.0)

    assert (
        validate_cosmos_videos(tmp_path, expected_frames=3, sample_rate_hz=15.0)
        == facts
    )


@pytest.mark.parametrize(
    ("mutation", "expected_frames", "message"),
    [
        (
            lambda root: (
                root / "fixed" / "clip_0000" / "rgb" / "rgb_0001.png"
            ).unlink(),
            3,
            "ordinal",
        ),
        (
            lambda root: (
                root / "wrist" / "clip_0000" / "edges" / "edges_0001.png"
            ).write_bytes(b"not a PNG"),
            3,
            "png|image",
        ),
        (lambda root: None, 4, "ordinal"),
        (
            lambda root: (
                root / "fixed" / "clip_0000" / "depth" / "depth_0002.png"
            ).unlink(),
            3,
            "ordinal",
        ),
    ],
)
def test_finalizer_rejects_incomplete_or_invalid_png_authority(
    tmp_path, mutation, expected_frames, message
):
    _write_png_sequences(tmp_path)
    mutation(tmp_path)

    with pytest.raises(ValueError, match=message):
        finalize_cosmos_videos(
            tmp_path, expected_frames=expected_frames, sample_rate_hz=15.0
        )


def test_validator_rejects_nonempty_undecodable_vendor_output(tmp_path):
    _write_png_sequences(tmp_path)
    finalize_cosmos_videos(tmp_path, expected_frames=3, sample_rate_hz=15.0)
    corrupt = tmp_path / "wrist" / "clip_0000" / "edges.mp4"
    corrupt.write_bytes(b"an empty H.264 access unit is not a video")

    with pytest.raises(ValueError, match="decode|video"):
        validate_cosmos_videos(tmp_path, expected_frames=3, sample_rate_hz=15.0)


def test_validator_rejects_video_with_wrong_rate(tmp_path):
    _write_png_sequences(tmp_path)
    finalize_cosmos_videos(tmp_path, expected_frames=3, sample_rate_hz=15.0)
    _write_h264_video(
        tmp_path / "fixed" / "clip_0000" / "rgb.mp4", frame_count=3, fps=12
    )

    with pytest.raises(ValueError, match="15|rate|FPS|fps"):
        validate_cosmos_videos(tmp_path, expected_frames=3, sample_rate_hz=15.0)
