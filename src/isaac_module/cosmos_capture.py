"""Finalize and validate CosmosWriter image sequences as deterministic videos."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

COSMOS_ROLES = ("fixed", "wrist")
SUPPORTED_COSMOS_ROLES = frozenset({"fixed", "fixed_2", "wrist"})
COSMOS_MODALITIES = ("rgb", "depth", "edges", "segmentation", "shaded_seg")


def _sample_rate(value: float) -> Fraction:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    return Fraction(str(rate))


def _capture_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def _expected_frame_count(expected_frames: int) -> int:
    if isinstance(expected_frames, bool) or not isinstance(expected_frames, int):
        raise TypeError("expected_frames must be an integer")
    if expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    return expected_frames


def _stage_start_frames(values: Sequence[int], frame_count: int) -> tuple[int, ...]:
    starts = tuple(values)
    if (
        not starts
        or starts[0] != 0
        or tuple(sorted(set(starts))) != starts
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in starts
        )
        or starts[-1] >= frame_count
    ):
        raise ValueError(
            "stage_start_frames must be unique increasing frame indices beginning at 0"
        )
    return starts


def _camera_roles(roles: Sequence[str]) -> tuple[str, ...]:
    result = tuple(roles)
    if (
        len(result) != len(set(result))
        or not {"fixed", "wrist"} <= set(result)
        or set(result) - SUPPORTED_COSMOS_ROLES
    ):
        raise ValueError(f"invalid Cosmos camera roles: {result!r}")
    return result


def _relative_video_path(role: str, modality: str) -> Path:
    return Path(role) / "clip_0000" / f"{modality}.mp4"


def _candidate_video_path(root: Path, role: str, modality: str) -> Path:
    clip_dir = root / role / "clip_0000"
    return clip_dir / f".{modality}.mp4.finalizing"


def _png_sequence(root: Path, role: str, modality: str, count: int) -> list[Path]:
    sequence_dir = root / role / "clip_0000" / modality
    if not sequence_dir.is_dir():
        raise ValueError(f"missing CosmosWriter image directory {sequence_dir}")

    pattern = re.compile(rf"{re.escape(modality)}_(\d{{4,}})\.png")
    indexed: dict[int, Path] = {}
    malformed: list[str] = []
    for path in sequence_dir.glob(f"{modality}_*.png"):
        match = pattern.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        ordinal = int(match.group(1))
        if path.name != f"{modality}_{ordinal:04d}.png":
            malformed.append(path.name)
            continue
        if ordinal in indexed:
            raise ValueError(
                f"duplicate {role}/{modality} PNG ordinal {ordinal}: "
                f"{indexed[ordinal].name}, {path.name}"
            )
        indexed[ordinal] = path

    expected_ordinals = set(range(count))
    actual_ordinals = set(indexed)
    if malformed or actual_ordinals != expected_ordinals:
        missing = sorted(expected_ordinals - actual_ordinals)
        unexpected = sorted(actual_ordinals - expected_ordinals)
        details = []
        if missing:
            details.append(f"missing ordinals {missing}")
        if unexpected:
            details.append(f"unexpected ordinals {unexpected}")
        if malformed:
            details.append(f"malformed names {sorted(malformed)}")
        raise ValueError(
            f"{role}/{modality} PNG frame sequence is not contiguous "
            f"0000..{count - 1:04d}: " + "; ".join(details)
        )
    return [indexed[ordinal] for ordinal in range(count)]


def _encode_sequence(
    paths: list[Path],
    candidate: Path,
    rate: Fraction,
    frame_time_base: Fraction,
    stage_start_frames: frozenset[int],
) -> None:
    width: int | None = None
    height: int | None = None
    try:
        # The explicit format is required because the diagnostic candidate does
        # not have an .mp4 suffix.
        with av.open(str(candidate), mode="w", format="mp4") as container:
            stream = container.add_stream(
                "libx264",
                rate=rate,
                options={"forced-idr": "1", "open-gop": "0"},
            )
            stream.pix_fmt = "yuv420p"
            stream.time_base = frame_time_base

            for ordinal, path in enumerate(paths):
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    frame_width, frame_height = rgb.size
                    if width is None:
                        width, height = frame_width, frame_height
                        if width % 2 or height % 2:
                            raise ValueError(
                                f"yuv420p requires even geometry, got {width}x{height} "
                                f"in {path}"
                            )
                        stream.width = width
                        stream.height = height
                    elif (frame_width, frame_height) != (width, height):
                        raise ValueError(
                            f"PNG geometry changed from {width}x{height} to "
                            f"{frame_width}x{frame_height} at {path}"
                        )
                    pixels = np.asarray(rgb, dtype=np.uint8)

                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = ordinal
                frame.time_base = frame_time_base
                if ordinal in stage_start_frames:
                    frame.pict_type = av.video.frame.PictureType.I
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:
        raise ValueError(
            f"failed to encode CosmosWriter candidate {candidate} from PNG sequence: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _as_fraction(value: object, label: str, relative_path: Path) -> Fraction:
    if value is None:
        raise ValueError(f"{relative_path.as_posix()} has no {label}")
    try:
        return Fraction(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            f"{relative_path.as_posix()} has invalid {label} {value!r}"
        ) from exc


def _decode_video(
    video_path: Path,
    relative_path: Path,
    expected_frames: int,
    rate: Fraction,
    frame_time_base: Fraction,
) -> dict[str, object]:
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise ValueError(f"missing or empty video {relative_path.as_posix()}")

    try:
        with av.open(str(video_path), mode="r") as container:
            all_streams = list(container.streams)
            streams = list(container.streams.video)
            if len(all_streams) != 1 or len(streams) != 1:
                raise ValueError(
                    f"expected exactly one video stream, found "
                    f"{len(all_streams)} total streams and {len(streams)} video streams"
                )
            stream = streams[0]
            codec = getattr(stream.codec_context, "name", None)
            if codec != "h264":
                raise ValueError(f"expected H.264 codec, found {codec!r}")

            average_rate = _as_fraction(
                stream.average_rate, "average frame rate", relative_path
            )
            if average_rate != rate:
                raise ValueError(
                    f"expected average frame rate {rate}, found {average_rate}"
                )

            stream_time_base = _as_fraction(
                stream.time_base, "stream time base", relative_path
            )
            stream_duration = stream.duration
            if stream_duration is None:
                raise ValueError("stream has no duration")
            duration = int(stream_duration) * stream_time_base
            expected_duration = Fraction(expected_frames, 1) / rate
            if duration != expected_duration:
                raise ValueError(
                    f"expected stream duration {expected_duration}, found {duration}"
                )

            container_duration_raw = container.duration
            if container_duration_raw is None:
                raise ValueError("container has no duration")
            container_duration = Fraction(int(container_duration_raw), av.time_base)
            expected_container_duration = Fraction(
                round(expected_duration * av.time_base), av.time_base
            )
            if container_duration != expected_container_duration:
                raise ValueError(
                    f"expected container duration {expected_container_duration}, "
                    f"found {container_duration}"
                )

            width: int | None = None
            height: int | None = None
            frame_count = 0
            previous_timestamp: Fraction | None = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f"decoded frame {frame_count} has no timestamp")
                timestamp = int(frame.pts) * Fraction(frame.time_base)
                expected_timestamp = frame_count * frame_time_base
                if timestamp != expected_timestamp:
                    raise ValueError(
                        f"decoded frame {frame_count} timestamp is {timestamp}, "
                        f"expected {expected_timestamp}"
                    )
                if (
                    previous_timestamp is not None
                    and timestamp - previous_timestamp != frame_time_base
                ):
                    raise ValueError(
                        f"decoded timestamp spacing is "
                        f"{timestamp - previous_timestamp}, expected {frame_time_base}"
                    )
                previous_timestamp = timestamp

                geometry = (int(frame.width), int(frame.height))
                if width is None:
                    width, height = geometry
                elif geometry != (width, height):
                    raise ValueError(
                        f"decoded geometry changed from {width}x{height} to "
                        f"{frame.width}x{frame.height}"
                    )
                pixel_format = getattr(frame.format, "name", None)
                if pixel_format != "yuv420p":
                    raise ValueError(
                        f"decoded frame {frame_count} has pixel format "
                        f"{pixel_format!r}, expected 'yuv420p'"
                    )
                frame_count += 1
    except ValueError as exc:
        raise ValueError(
            f"invalid Cosmos video {relative_path.as_posix()}: {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"cannot decode {relative_path.as_posix()}: {type(exc).__name__}: {exc}"
        ) from exc

    if frame_count != expected_frames:
        raise ValueError(
            f"invalid Cosmos video {relative_path.as_posix()}: expected "
            f"{expected_frames} decoded frames, found {frame_count}"
        )
    if width is None or height is None:
        raise ValueError(
            f"invalid Cosmos video {relative_path.as_posix()}: no decoded frames"
        )

    return {
        "relative_path": relative_path.as_posix(),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "average_fps": float(rate),
        "raw_average_fps": float(rate),
        "timestamp_spacing_s": float(frame_time_base),
        "duration_s": float(Fraction(expected_frames, 1) / rate),
    }


def _validate_paths(
    root: Path,
    expected_frames: int,
    path_for: Callable[[str, str], Path],
    camera_roles: Sequence[str],
    rate: Fraction,
    frame_time_base: Fraction,
) -> dict[str, object]:
    roles: dict[str, dict[str, dict[str, object]]] = {}
    for role in camera_roles:
        modalities: dict[str, dict[str, object]] = {}
        expected_geometry: tuple[int, int] | None = None
        for modality in COSMOS_MODALITIES:
            relative_path = _relative_video_path(role, modality)
            facts = _decode_video(
                path_for(role, modality),
                relative_path,
                expected_frames,
                rate,
                frame_time_base,
            )
            width = facts["width"]
            height = facts["height"]
            if not isinstance(width, int) or not isinstance(height, int):
                raise TypeError(
                    f"decoded video facts for {relative_path.as_posix()} must have "
                    "integer width and height"
                )
            geometry = (width, height)
            if expected_geometry is None:
                expected_geometry = geometry
            elif geometry != expected_geometry:
                raise ValueError(
                    f"{role} modalities have inconsistent geometry: expected "
                    f"{expected_geometry[0]}x{expected_geometry[1]}, {modality} has "
                    f"{geometry[0]}x{geometry[1]}"
                )
            modalities[modality] = facts
        roles[role] = modalities
    return {"path": str(root), "roles": roles}


def validate_cosmos_videos(
    root: str | Path,
    expected_frames: int,
    sample_rate_hz: float,
    roles: Sequence[str] = COSMOS_ROLES,
) -> dict[str, object]:
    """Fully validate finalized Cosmos videos and return JSON-safe facts."""

    capture_root = _capture_root(root)
    count = _expected_frame_count(expected_frames)
    camera_roles = _camera_roles(roles)
    rate = _sample_rate(sample_rate_hz)
    frame_time_base = 1 / rate
    return _validate_paths(
        capture_root,
        count,
        lambda role, modality: capture_root / _relative_video_path(role, modality),
        camera_roles,
        rate,
        frame_time_base,
    )


def finalize_cosmos_videos(
    root: str | Path,
    expected_frames: int,
    sample_rate_hz: float,
    roles: Sequence[str] = COSMOS_ROLES,
    stage_start_frames: Sequence[int] = (0,),
) -> dict[str, object]:
    """Encode PNG sequences, validate them, then replace vendor MP4s."""

    capture_root = _capture_root(root)
    count = _expected_frame_count(expected_frames)
    camera_roles = _camera_roles(roles)
    rate = _sample_rate(sample_rate_hz)
    frame_time_base = 1 / rate
    keyframes = frozenset(_stage_start_frames(stage_start_frames, count))
    candidates: dict[tuple[str, str], Path] = {}

    for role in camera_roles:
        for modality in COSMOS_MODALITIES:
            candidate = _candidate_video_path(capture_root, role, modality)
            paths = _png_sequence(capture_root, role, modality, count)
            _encode_sequence(paths, candidate, rate, frame_time_base, keyframes)
            candidates[(role, modality)] = candidate

    _validate_paths(
        capture_root,
        count,
        lambda role, modality: candidates[(role, modality)],
        camera_roles,
        rate,
        frame_time_base,
    )

    for role in camera_roles:
        for modality in COSMOS_MODALITIES:
            destination = capture_root / _relative_video_path(role, modality)
            os.replace(candidates[(role, modality)], destination)

    return validate_cosmos_videos(
        capture_root,
        count,
        sample_rate_hz=sample_rate_hz,
        roles=camera_roles,
    )
