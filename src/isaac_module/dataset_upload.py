"""Upload finalized episode directories to a Viam dataset as .zip files.

Runs inside the module, next to the episode files. Credentials and the machine
part ID come from the environment viam-server provides to every module.

Episode directories are deleted after a confirmed `file_id` lands, matching the
data-manager file-sync behavior the module replaced. Upload failures keep the
files on disk for retry.
"""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from viam.app.data_client import DataClient
from viam.app.viam_client import ViamClient

UPLOAD_ATTEMPTS = 3
UPLOAD_RETRY_DELAYS_SEC = (2.0, 5.0)
_PRECOMPRESSED_SUFFIXES = frozenset({".mp4", ".parquet"})


def zip_episode(episode_dir: Path) -> bytes:
    """Zip an episode directory under its own name, storing media uncompressed."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        for path in sorted(episode_dir.rglob("*")):
            if not path.is_file():
                continue
            compression = (
                zipfile.ZIP_STORED
                if path.suffix in _PRECOMPRESSED_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            relative = path.relative_to(episode_dir).as_posix()
            archive.write(
                path,
                arcname=f"{episode_dir.name}/{relative}",
                compress_type=compression,
            )
    return buffer.getvalue()


@dataclass(frozen=True)
class UploadRequest:
    dataset_id: str
    tags: tuple[str, ...] = ()
    app_url: Optional[str] = None
    cosmos_path: Optional[str] = None

    @classmethod
    def from_command(cls, value: Any) -> "UploadRequest":
        if not isinstance(value, Mapping) or not value.get("dataset_id"):
            raise ValueError("upload requires a dataset_id")
        tags = value.get("tags") or []
        return cls(
            dataset_id=str(value["dataset_id"]),
            tags=tuple(str(tag) for tag in tags),
            app_url=str(value["app_url"]) if value.get("app_url") else None,
            cosmos_path=str(value["cosmos_path"]) if value.get("cosmos_path") else None,
        )


async def _file_upload(
    data_client: DataClient,
    part_id: str,
    episode_dir: Path,
    payload: bytes,
    request: UploadRequest,
    component_name: str,
) -> str:
    return await data_client.file_upload(
        part_id=part_id,
        data=payload,
        component_type="generic",
        component_name=component_name,
        file_name=f"episode-{episode_dir.name}",
        file_extension=".zip",
        mime_type="application/zip",
        tags=list(request.tags),
        dataset_ids=[request.dataset_id],
    )


async def _connect(app_url: Optional[str]) -> ViamClient:
    return await ViamClient.create_from_env_vars(app_url=app_url)


def _cleanup_after_upload(
    result: dict[str, Any], episode_dir: Path, cosmos_path: Optional[str]
) -> None:
    """Delete finalized episode output after a confirmed cloud upload."""
    try:
        shutil.rmtree(episode_dir)
        result["deleted"] = True
    except OSError as exc:
        result["delete_error"] = f"{type(exc).__name__}: {exc}"
    if cosmos_path:
        cosmos_dir = Path(cosmos_path)
        if not cosmos_dir.exists():
            return
        try:
            shutil.rmtree(cosmos_dir)
            result["cosmos_deleted"] = True
        except OSError as exc:
            result["cosmos_delete_error"] = f"{type(exc).__name__}: {exc}"


async def upload_episode(
    episode_dir: Path,
    request: UploadRequest,
    component_name: str,
    *,
    connect: Callable[[Optional[str]], Awaitable[ViamClient]] = _connect,
    retry_delays_sec: Sequence[float] = UPLOAD_RETRY_DELAYS_SEC,
) -> dict[str, Any]:
    """Upload one episode, retrying transient failures.

    Never raises: the episode is already finalized on disk, so a failed upload
    is reported in the result for the caller to retry later.
    """
    result: dict[str, Any] = {"dataset_id": request.dataset_id}
    try:
        if not (episode_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"no finalized episode at {episode_dir}")
        part_id = os.environ.get("VIAM_MACHINE_PART_ID")
        if not part_id:
            raise RuntimeError("VIAM_MACHINE_PART_ID is not set")
        payload = await asyncio.to_thread(zip_episode, episode_dir)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    delays = list(retry_delays_sec)
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        client = None
        try:
            client = await connect(request.app_url)
            file_id = await _file_upload(
                client.data_client,
                part_id,
                episode_dir,
                payload,
                request,
                component_name,
            )
            result.pop("error", None)
            result.update(file_id=file_id, bytes=len(payload), attempts=attempt)
            _cleanup_after_upload(result, episode_dir, request.cosmos_path)
            return result
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["attempts"] = attempt
        finally:
            if client is not None:
                client.close()
        if attempt < UPLOAD_ATTEMPTS and delays:
            await asyncio.sleep(delays[min(attempt - 1, len(delays) - 1)])
    return result
