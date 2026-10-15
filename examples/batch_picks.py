"""Generate a serial, resumable run of synchronized scripted pick episodes."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm
from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.services.motion import MotionClient

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_module.cosmos_capture import validate_cosmos_videos

if __package__:
    from .scripted_pick import (
        HOME_JOINT_DEGREES,
        DEFAULT_PROMPT,
        EpisodeConfig,
        PickConfig,
        execute_pick,
        resolve_pick_region,
    )
else:
    from scripted_pick import (
        HOME_JOINT_DEGREES,
        DEFAULT_PROMPT,
        EpisodeConfig,
        PickConfig,
        execute_pick,
        resolve_pick_region,
    )

RUN_SCHEMA_VERSION = 5
DEFAULT_NOISE_FRACTION = 0.6


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")




def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _noise_fraction(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("noise_fraction must be numeric")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("noise_fraction must be finite and between 0 and 1")
    return fraction


def _episode_uses_noise(seed: int, noise_fraction: float) -> bool:
    return random.Random(f"episode-noise:{int(seed)}").random() < noise_fraction


def _run_config(args: argparse.Namespace, pick_config: PickConfig) -> dict[str, Any]:
    config = {
        "address": args.address,
        "count": args.count,
        "start_seed": args.start_seed,
        "task": args.task,
        "world": args.world,
        "gripper": args.gripper,
        "motion": args.motion,
        "prop": args.prop,
        "arm": args.arm,
        "fixed_camera": args.fixed_camera,
        "fixed_camera_2": args.fixed_camera_2,
        "wrist_camera": args.wrist_camera,
        "grasp_height_m": args.grasp_height_m,
        "yaw_offset_deg": args.yaw_offset_deg,
        "gripper_settle_sec": args.gripper_settle_sec,
        "noise_fraction": _noise_fraction(args.noise_fraction),
        "pick_region_min_m": list(pick_config.pick_region_min_m),
        "pick_region_max_m": list(pick_config.pick_region_max_m),
        "cosmos_output": bool(args.cosmos_output),
    }
    if pick_config.grip_only:
        config.update(grip_only=True, grip_lift_m=pick_config.grip_lift_m)
    nudge = _nudge(args)
    if nudge is not None:
        config["nudge"] = nudge
    if args.colored_blocks:
        config["colored_blocks"] = dict(args.colored_blocks)
    if args.place_on:
        config["place_on"] = dict(args.place_on)
    if args.park_props:
        config["park_props"] = list(args.park_props)
    return config


def _colored_blocks(value: str) -> dict[str, str]:
    """Parse --colored-blocks: two label=prop pairs, e.g. red=pick_block,blue=blue_block."""
    blocks: dict[str, str] = {}
    for part in value.split(","):
        label, _, prop = part.partition("=")
        label, prop = label.strip(), prop.strip()
        if not label or not prop or label in blocks:
            raise argparse.ArgumentTypeError(
                "--colored-blocks takes two label=prop pairs, e.g. "
                "red=pick_block,blue=blue_block"
            )
        blocks[label] = prop
    if len(blocks) != 2 or len(set(blocks.values())) != 2:
        raise argparse.ArgumentTypeError(
            "--colored-blocks takes two label=prop pairs with different props"
        )
    return blocks


def _target_block(seed: int, blocks: Mapping[str, str]) -> tuple[str, str, str]:
    """Pick the seed's target (label, prop) and the other block as distractor."""
    labels = sorted(blocks)
    label = random.Random(f"target-block:{int(seed)}").choice(labels)
    (other,) = [candidate for candidate in labels if candidate != label]
    return label, blocks[label], blocks[other]


def _place_on(value: str) -> dict[str, str]:
    """Parse --place-on: one label=prop pair, e.g. green=place_pad."""
    label, _, prop = value.partition("=")
    label, prop = label.strip(), prop.strip()
    if not label or not prop or "," in value:
        raise argparse.ArgumentTypeError(
            "--place-on takes one label=prop pair, e.g. green=place_pad"
        )
    return {label: prop}


def _nudge_deg(value: str) -> float | list[float]:
    """Parse --nudge-deg: one value for every joint, or six comma-separated."""
    try:
        parts = [float(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--nudge-deg takes a number or six comma-separated numbers"
        ) from None
    if len(parts) not in (1, 6):
        raise argparse.ArgumentTypeError(
            "--nudge-deg takes a number or six comma-separated numbers"
        )
    if not all(math.isfinite(part) and part >= 0 for part in parts):
        raise argparse.ArgumentTypeError("--nudge-deg values must be non-negative")
    return parts[0] if len(parts) == 1 else parts


def _nudge(args: argparse.Namespace) -> dict[str, float | list[float]] | None:
    amplitude = args.nudge_deg
    if not (max(amplitude) if isinstance(amplitude, list) else amplitude):
        return None
    return {
        "amplitude_deg": args.nudge_deg,
        "interval_s": args.nudge_interval_sec,
        "duration_s": args.nudge_duration_sec,
    }


def _git_sha() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    sha = result.stdout.strip()
    if not sha:
        raise RuntimeError("git rev-parse HEAD returned an empty SHA")
    return sha


def _load_or_create_run(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes").mkdir(exist_ok=True)
    run_path = run_dir / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("schema_version") != RUN_SCHEMA_VERSION:
            raise ValueError(f"unsupported run schema_version in {run_path}")
        stored_config = run.get("config")
        if stored_config != config:
            if not isinstance(stored_config, dict):
                raise ValueError("resume configuration does not match run.json")
            stored_count = stored_config.get("count")
            requested_count = config.get("count")
            comparable_config = dict(config)
            comparable_config["count"] = stored_count
            if (
                stored_config != comparable_config
                or not isinstance(stored_count, int)
                or not isinstance(requested_count, int)
                or requested_count <= stored_count
            ):
                raise ValueError("resume configuration does not match run.json")
            stored_config["count"] = requested_count
            run["updated_at"] = _utc_now()
            _write_json_atomic(run_path, run)
        return run

    run = {
        "schema_version": RUN_SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config": dict(config),
        "attempts": [],
    }
    _write_json_atomic(run_path, run)
    return run


def _episode_paths(
    run_dir: Path, episode_id: str, cosmos_output: bool
) -> tuple[Path, Path | None]:
    native_episode_dir = run_dir / "episodes" / episode_id
    cosmos_episode_dir = (
        run_dir / "cosmoswriter" / episode_id if cosmos_output else None
    )
    return native_episode_dir, cosmos_episode_dir


def _attempt_timing(
    started_at: str, started_monotonic: float, home_move_sec: float
) -> dict[str, Any]:
    """Additive run.json fields: wall time for the whole attempt (home move,
    pick, capture finalization) so per-episode throughput can be measured.
    Per-phase detail lives in attempt["result"]["timing"] from execute_pick."""
    return {
        "started_at": started_at,
        "wall_time_sec": time.monotonic() - started_monotonic,
        "home_move_sec": home_move_sec,
    }


def _record_attempt(run: dict[str, Any], attempt: Mapping[str, Any]) -> None:
    attempts = run["attempts"]
    attempts[:] = [
        item for item in attempts if item["episode_id"] != attempt["episode_id"]
    ]
    attempts.append(dict(attempt))
    attempts.sort(key=lambda item: item["episode_index"])
    run["updated_at"] = _utc_now()


def _validate_cosmoswriter(
    episode_dir: Path,
    native_sample_count: int,
    sample_rate_hz: float,
    roles: tuple[str, ...],
) -> dict[str, Any]:
    return validate_cosmos_videos(
        episode_dir,
        native_sample_count,
        sample_rate_hz=sample_rate_hz,
        roles=roles,
    )


def _remote_attempt(
    episode_index: int, seed: int, noise: bool, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Record an episode whose files were written on the sim machine.

    When batch_picks runs on another computer the episode folder is not
    local, so the outcome comes from execute_pick and the path from the
    sim's stop_episode reply."""
    capture = result.get("capture") or {}
    return {
        "episode_index": episode_index,
        "episode_id": f"{episode_index:06d}",
        "seed": seed,
        "noise": noise,
        "status": "remote",
        "outcome": result["outcome"],
        "sample_count": capture.get("sample_count"),
        "path": capture.get("path"),
    }


def _finished_remote_attempt(
    run: Mapping[str, Any], episode_id: str, seed: int
) -> dict[str, Any] | None:
    """A previously recorded remote episode to keep instead of re-recording."""
    for attempt in run["attempts"]:
        if (
            attempt.get("episode_id") == episode_id
            and attempt.get("seed") == seed
            and attempt.get("status") == "remote"
            and attempt.get("outcome") in ("success", "failure")
        ):
            return dict(attempt)
    return None


async def _close_open_episode(world: Any) -> None:
    try:
        await asyncio.wait_for(
            world.do_command({"command": "stop_episode", "outcome": "incomplete"}),
            timeout=60,
        )
    except Exception:
        # Best effort: usually the cancelled pick already closed the capture,
        # so this second stop fails and there is nothing left to do.
        pass


def _props_to_park(
    park_props: Sequence[str], config: PickConfig
) -> list[str]:
    """Props listed with --park-props that this episode does not place."""
    used = {config.prop_name, config.distractor_prop, config.place_prop}
    return [prop for prop in park_props if prop not in used]


def _finalized_attempt(
    episode_index: int,
    seed: int,
    noise: bool,
    episode_dir: Path,
    cosmos_episode_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("episode_id") != episode_dir.name or manifest.get("seed") != seed:
        raise ValueError(f"finalized episode metadata does not match {episode_dir}")
    attempt = {
        "episode_index": episode_index,
        "episode_id": episode_dir.name,
        "seed": seed,
        "noise": noise,
        "status": "finalized",
        "outcome": manifest["outcome"],
        "sample_count": manifest["sample_count"],
        "path": str(episode_dir),
    }
    if cosmos_episode_dir is not None:
        cameras = manifest.get("cameras", [])
        roles = tuple(
            camera["role"] for camera in cameras if isinstance(camera, dict)
        ) or ("fixed", "wrist")
        try:
            attempt["cosmoswriter"] = _validate_cosmoswriter(
                cosmos_episode_dir,
                manifest["sample_count"],
                manifest["sample_rate_hz"],
                roles,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            attempt.update(
                status="error",
                outcome="incomplete",
                native_outcome=manifest["outcome"],
                error=(
                    f"CosmosWriter validation failed at {cosmos_episode_dir}: "
                    f"{type(exc).__name__}: {exc}. Native episode {episode_dir} "
                    "remains finalized and was not recaptured; inspect or repair the "
                    "CosmosWriter output before resuming"
                ),
            )
    return attempt


def _module_attempt(
    episode_index: int,
    seed: int,
    noise: bool,
    episode_id: str,
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a finalized attempt from the module's stop_episode result.

    Used when the episode was written on the machine and is not visible on this
    host's filesystem.
    """
    if (
        capture.get("episode_id") != episode_id
        or not capture.get("path")
        or "sample_count" not in capture
    ):
        raise ValueError(f"module did not report finalized episode {episode_id}")
    attempt = {
        "episode_index": episode_index,
        "episode_id": episode_id,
        "seed": seed,
        "noise": noise,
        "status": "finalized",
        "outcome": str(capture["outcome"]),
        "sample_count": int(capture["sample_count"]),
        "path": str(capture["path"]),
        # Resume trusts run.json for episodes stored only on the machine.
        "on_machine": True,
    }
    cosmos_output_dir = capture.get("cosmos_output_dir")
    if cosmos_output_dir:
        attempt["cosmos_path"] = str(cosmos_output_dir)
    return attempt


def _upload_request(args: argparse.Namespace, run_dir: Path) -> dict[str, Any] | None:
    if not args.upload_dataset_id:
        return None
    request: dict[str, Any] = {
        "dataset_id": args.upload_dataset_id,
        "tags": [run_dir.name, "isaac-episode", *args.upload_tag],
    }
    if args.app_url:
        request["app_url"] = args.app_url
    return request


def _upload_record(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    record = dict(value)
    # DoCommand results arrive as protobuf Structs, which turn ints into floats.
    for key in ("bytes", "attempts"):
        if isinstance(record.get(key), float):
            record[key] = int(record[key])
    return record


def _uploaded_to(record: Mapping[str, Any] | None, dataset_id: str) -> bool:
    return (
        record is not None
        and bool(record.get("file_id"))
        and record.get("dataset_id") == dataset_id
    )


async def generate_run(
    args: argparse.Namespace,
    world: Any,
    gripper: Any,
    motion: Any,
    arm: Any,
    pick_config: PickConfig | None = None,
) -> dict[str, Any]:
    noise_fraction = _noise_fraction(args.noise_fraction)
    pick_config = pick_config or PickConfig(
        prop_name=args.prop,
        arm_name=args.arm,
        gripper_name=args.gripper,
        grasp_height_m=args.grasp_height_m,
        yaw_offset_deg=args.yaw_offset_deg,
        gripper_settle_sec=args.gripper_settle_sec,
        grip_only=args.grip_only,
        grip_lift_m=args.grip_lift_m,
    )
    run_dir = Path(args.run_dir).expanduser().resolve()
    run = _load_or_create_run(run_dir, _run_config(args, pick_config))
    run_path = run_dir / "run.json"
    previous_attempts = {attempt["episode_id"]: attempt for attempt in run["attempts"]}
    upload = _upload_request(args, run_dir)
    episodes_dir = run_dir / "episodes"
    for episode_index in tqdm(range(args.count), desc="Episodes", unit="episode"):
        episode_id = f"{episode_index:06d}"
        seed = args.start_seed + episode_index
        episode_noise = _episode_uses_noise(seed, noise_fraction)
        episode_pick_config = replace(pick_config, noise=episode_noise)
        episode_task = args.task
        target_label = ""
        if args.colored_blocks:
            target_label, target_prop, distractor_prop = _target_block(
                seed, args.colored_blocks
            )
            episode_pick_config = replace(
                episode_pick_config,
                prop_name=target_prop,
                distractor_prop=distractor_prop,
                target_label=target_label,
            )
            episode_task = f"pick up the {target_label} block"
        if args.place_on:
            ((place_label, place_prop),) = args.place_on.items()
            episode_pick_config = replace(
                episode_pick_config, place_prop=place_prop, place_label=place_label
            )
            episode_task = f"put the block on the {place_label} pad"
        final_dir, cosmos_episode_dir = _episode_paths(
            run_dir, episode_id, args.cosmos_output
        )
        partial_dir = episodes_dir / f".{episode_id}.partial"
        previous = previous_attempts.get(episode_id)
        captured_now = False

        previous_remote = _finished_remote_attempt(run, episode_id, seed)
        if final_dir.is_dir():
            attempt = _finalized_attempt(
                episode_index, seed, episode_noise, final_dir, cosmos_episode_dir
            )
        elif (
            previous is not None
            and previous.get("on_machine")
            and previous.get("status") == "finalized"
        ):
            attempt = dict(previous)
        elif previous_remote is not None:
            attempt = previous_remote
        elif partial_dir.exists():
            error = "partial episode requires operator inspection"
            if cosmos_episode_dir is not None and cosmos_episode_dir.exists():
                error += f"; CosmosWriter output at {cosmos_episode_dir} was preserved"
            attempt = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "seed": seed,
                "noise": episode_noise,
                "status": "partial",
                "outcome": "incomplete",
                "path": str(partial_dir),
                "error": error,
            }
        elif cosmos_episode_dir is not None and cosmos_episode_dir.exists():
            attempt = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "seed": seed,
                "noise": episode_noise,
                "status": "error",
                "outcome": "incomplete",
                "path": None,
                "error": (
                    f"CosmosWriter output already exists at {cosmos_episode_dir} while "
                    "the native episode is missing; inspect or remove it explicitly. "
                    "Capture was not run and existing output was not overwritten"
                ),
            }
        else:
            started_at = _utc_now()
            attempt_started = time.monotonic()
            home_move_sec = 0.0
            try:
                parked = _props_to_park(args.park_props, episode_pick_config)
                if parked:
                    await world.do_command({"command": "park_props", "props": parked})
                if not episode_pick_config.grip_only:
                    home_started = time.monotonic()
                    await arm.move_to_joint_positions(
                        JointPositions(values=list(HOME_JOINT_DEGREES))
                    )
                    home_move_sec = time.monotonic() - home_started
                pick = execute_pick(
                    world,
                    gripper,
                    motion,
                    arm,
                    seed,
                    episode_pick_config,
                    EpisodeConfig(
                        output_dir=str(episodes_dir),
                        episode_id=episode_id,
                        task=episode_task,
                        arm_name=args.arm,
                        fixed_camera_name=args.fixed_camera,
                        wrist_camera_name=args.wrist_camera,
                        fixed_camera_2_name=args.fixed_camera_2,
                        cosmos_output_dir=(
                            str(cosmos_episode_dir)
                            if cosmos_episode_dir is not None
                            else None
                        ),
                        upload=upload,
                        nudge=_nudge(args),
                    ),
                )
                if args.episode_timeout_sec > 0:
                    try:
                        result = await asyncio.wait_for(
                            pick, timeout=args.episode_timeout_sec
                        )
                    except TimeoutError:
                        # A hung call can leave capture running on the sim,
                        # which would block every later episode.
                        await _close_open_episode(world)
                        raise
                else:
                    result = await pick
                captured_now = True
                capture = result.get("capture") or {}
                if final_dir.is_dir():
                    attempt = _finalized_attempt(
                        episode_index,
                        seed,
                        episode_noise,
                        final_dir,
                        cosmos_episode_dir,
                    )
                else:
                    attempt = _module_attempt(
                        episode_index, seed, episode_noise, episode_id, capture
                    )
                attempt["result"] = dict(result)
                upload_record = _upload_record(capture.get("upload"))
                if upload_record is not None:
                    attempt["upload"] = upload_record
            except Exception as exc:
                if final_dir.is_dir():
                    attempt = _finalized_attempt(
                        episode_index, seed, episode_noise, final_dir, cosmos_episode_dir
                    )
                else:
                    error = f"{type(exc).__name__}: {exc}"
                    if cosmos_episode_dir is not None and cosmos_episode_dir.exists():
                        error += (
                            f"; CosmosWriter output at {cosmos_episode_dir} was "
                            "preserved and requires operator inspection"
                        )
                    attempt = {
                        "episode_index": episode_index,
                        "episode_id": episode_id,
                        "seed": seed,
                        "noise": episode_noise,
                        "status": "partial" if partial_dir.exists() else "error",
                        "outcome": "incomplete",
                        "path": str(partial_dir) if partial_dir.exists() else None,
                        "error": error,
                    }
            attempt.update(
                _attempt_timing(started_at, attempt_started, home_move_sec)
            )

        if attempt["status"] == "finalized" and "upload" not in attempt:
            previous_upload = _upload_record((previous or {}).get("upload"))
            if previous_upload is not None:
                attempt["upload"] = previous_upload
        if (
            attempt["status"] == "finalized"
            and upload is not None
            and not captured_now
            and not _uploaded_to(attempt.get("upload"), upload["dataset_id"])
        ):
            # The module already retried during capture; this retries uploads
            # that failed in an earlier invocation, or uploads to a new dataset.
            upload_payload = dict(upload)
            if attempt.get("cosmos_path") and not upload_payload.get("cosmos_path"):
                upload_payload["cosmos_path"] = attempt["cosmos_path"]
            try:
                attempt["upload"] = _upload_record(
                    await world.do_command(
                        {
                            "command": "upload_episode",
                            "path": attempt["path"],
                            "upload": upload_payload,
                        }
                    )
                )
            except Exception as exc:
                attempt["upload"] = {
                    "dataset_id": upload["dataset_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }

        if target_label:
            attempt["target_block"] = target_label
        _record_attempt(run, attempt)
        _write_json_atomic(run_path, run)

    counts: dict[str, int] = {}
    for attempt in run["attempts"]:
        key = str(attempt["outcome"])
        counts[key] = counts.get(key, 0) + 1
    result = {
        "run_dir": str(run_dir),
        "counts": counts,
        "attempts": run["attempts"],
    }
    upload_errors = [
        attempt["episode_id"]
        for attempt in run["attempts"]
        if "error" in attempt.get("upload", {})
    ]
    if upload_errors:
        result["upload_errors"] = upload_errors
    return result


async def _main(args: argparse.Namespace) -> Mapping[str, Any]:
    api_key = os.environ.get("VIAM_API_KEY")
    api_key_id = os.environ.get("VIAM_API_KEY_ID")
    if bool(api_key) != bool(api_key_id):
        raise ValueError("set both VIAM_API_KEY and VIAM_API_KEY_ID, or neither")
    options = (
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
        if api_key and api_key_id
        else RobotClient.Options(dial_options=DialOptions(insecure=True))
    )
    if args.disable_webrtc:
        assert options.dial_options is not None
        options.dial_options.disable_webrtc = True

    async with await RobotClient.at_address(args.address, options) as machine:
        pick_region_min_m, pick_region_max_m = await resolve_pick_region(
            machine, args.arm
        )
        return await generate_run(
            args,
            Generic.from_robot(machine, args.world),
            Gripper.from_robot(machine, args.gripper),
            MotionClient.from_robot(machine, args.motion),
            Arm.from_robot(machine, args.arm),
            PickConfig(
                prop_name=args.prop,
                arm_name=args.arm,
                gripper_name=args.gripper,
                grasp_height_m=args.grasp_height_m,
                yaw_offset_deg=args.yaw_offset_deg,
                gripper_settle_sec=args.gripper_settle_sec,
                grip_only=args.grip_only,
                grip_lift_m=args.grip_lift_m,
                pick_region_min_m=pick_region_min_m,
                pick_region_max_m=pick_region_max_m,
            ),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="machine address from the Viam CONNECT tab")
    parser.add_argument(
        "run_dir", help="run directory containing run.json and episodes/"
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_PROMPT)
    parser.add_argument("--world", default="sim-world")
    parser.add_argument("--gripper", default="pick-gripper")
    parser.add_argument("--motion", default="builtin")
    parser.add_argument("--prop", default="pick_block")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--fixed-camera", default="scene-cam")
    parser.add_argument(
        "--fixed-camera-2",
        default="",
        help="optional second fixed camera component; empty disables it",
    )
    parser.add_argument("--wrist-camera", default="wrist-cam")
    parser.add_argument(
        "--grasp-height-m", type=float, default=PickConfig.grasp_height_m
    )
    parser.add_argument(
        "--gripper-settle-sec",
        type=float,
        default=PickConfig.gripper_settle_sec,
        help="maximum simulator seconds to wait for a firm grasp",
    )
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=DEFAULT_NOISE_FRACTION,
        help="fraction of episodes that receive seeded action noise (default: %(default)s)",
    )
    parser.add_argument(
        "--grip-only",
        action="store_true",
        help="capture only the pre-pick descent, grip, and short lift",
    )
    parser.add_argument(
        "--grip-lift-m",
        type=float,
        default=PickConfig.grip_lift_m,
        help="vertical lift distance for --grip-only (default: %(default)s m)",
    )
    parser.add_argument(
        "--nudge-deg",
        type=_nudge_deg,
        default=0.0,
        help=(
            "nudge each joint by up to +/- this many degrees while recording "
            "(0: off); one value, or six comma-separated per-joint values"
        ),
    )
    parser.add_argument(
        "--nudge-interval-sec",
        type=float,
        default=3.0,
        help="mean seconds between nudge starts",
    )
    parser.add_argument(
        "--nudge-duration-sec",
        type=float,
        default=0.6,
        help="seconds each nudge takes to fade in and back out",
    )
    parser.add_argument(
        "--colored-blocks",
        type=_colored_blocks,
        default=None,
        help=(
            "two label=prop pairs (e.g. red=pick_block,blue=blue_block): each "
            "episode places both blocks, picks one by seed, and uses the task "
            "'pick up the <label> block'"
        ),
    )
    parser.add_argument(
        "--place-on",
        type=_place_on,
        default=None,
        help=(
            "label=prop for a flat pad (e.g. green=place_pad): each episode "
            "places the pad, picks the block, puts it on the pad, and uses the "
            "task 'put the block on the <label> pad'"
        ),
    )
    parser.add_argument(
        "--park-props",
        type=lambda value: [part.strip() for part in value.split(",") if part.strip()],
        default=[],
        help=(
            "comma-separated props to move off the table before each episode "
            "unless that episode uses them (e.g. blue_block,place_pad)"
        ),
    )
    parser.add_argument(
        "--remote-episodes",
        action="store_true",
        help=(
            "the sim writes episodes on another machine: record each outcome "
            "from the sim's reply instead of reading the local episode folder, "
            "and skip those episodes when the run is resumed"
        ),
    )
    parser.add_argument(
        "--episode-timeout-sec",
        type=float,
        default=0.0,
        help=(
            "give up on an episode after this many seconds, close its capture, "
            "and continue (0 = no limit)"
        ),
    )
    parser.add_argument("--disable-webrtc", action="store_true")
    parser.add_argument(
        "--cosmos-output",
        action="store_true",
        help="also write episode-keyed CosmosWriter output beside episodes/",
    )
    parser.add_argument(
        "--upload-dataset-id",
        help="have the machine upload each finalized episode as a .zip to this "
        "Viam dataset",
    )
    parser.add_argument(
        "--upload-tag",
        action="append",
        default=[],
        help="extra tag for uploaded episodes; may be repeated",
    )
    parser.add_argument(
        "--app-url",
        help="Viam app host the machine uploads to (default: the SDK default, "
        "app.viam.com)",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    try:
        args.noise_fraction = _noise_fraction(args.noise_fraction)
    except ValueError as exc:
        parser.error(str(exc))
    if args.gripper_settle_sec < 0:
        parser.error("--gripper-settle-sec must be non-negative")
    if args.grip_lift_m <= 0:
        parser.error("--grip-lift-m must be positive")
    if args.nudge_interval_sec <= 0 or args.nudge_duration_sec <= 0:
        parser.error("--nudge-interval-sec and --nudge-duration-sec must be positive")
    if args.place_on and (args.colored_blocks or args.grip_only):
        parser.error("--place-on can't be combined with --colored-blocks or --grip-only")
    return args


def main() -> int:
    result = asyncio.run(_main(_parse_args()))
    print(json.dumps(result, sort_keys=True))
    ok = set(result["counts"]) <= {"success", "failure"}
    return 0 if ok and not result.get("upload_errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
