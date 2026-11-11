"""Serves a model's custom control artifact, declared in the module's own meta.json, over do_command.

The command is transitional: keep the `custom_control` entries in meta.json and delete this file
once the platform serves the declared artifact itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CUSTOM_CONTROL_COMMAND = "get-custom-control"

_MODELS_KEY = "models"
_MODEL_KEY = "model"
_CUSTOM_CONTROL_KEY = "custom_control"

_cache: dict[tuple[Path, str], dict[str, str]] = {}


class CustomControlError(Exception):
    pass


def _read_manifest(meta_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CustomControlError(f"meta.json not found at {meta_path}") from error
    except json.JSONDecodeError as error:
        raise CustomControlError(f"meta.json at {meta_path} is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise CustomControlError(f"meta.json at {meta_path} must contain a JSON object")
    return manifest


def _declared_path(model: str, meta_path: Path) -> str:
    entries = _read_manifest(meta_path).get(_MODELS_KEY, [])
    entry = next(
        (e for e in entries if isinstance(e, dict) and e.get(_MODEL_KEY) == model), None
    )
    if entry is None:
        raise CustomControlError(f"model {model!r} has no entry in models of {meta_path}")
    relative_path = entry.get(_CUSTOM_CONTROL_KEY)
    if not relative_path:
        raise CustomControlError(
            f"model {model!r} has no custom_control in {meta_path}"
        )
    return relative_path


def load_custom_control(model: object, meta_path: Path) -> dict[str, str]:
    """Return {"html", "revision", "path"} (path as written in meta.json), reading the artifact once per (meta_path, model).

    `model` is matched against meta.json by its string form, so the SDK's `Model` object
    (what `EasyResource` stores on `MODEL`) and the plain "ns:family:name" triplet both work.
    """
    model = str(model)
    key = (meta_path, model)
    if key not in _cache:
        declared_path = _declared_path(model, meta_path)
        artifact_path = meta_path.parent / declared_path
        try:
            artifact_bytes = artifact_path.read_bytes()
        except FileNotFoundError as error:
            raise CustomControlError(
                f"custom_control file for model {model!r} not found at {artifact_path}"
            ) from error
        _cache[key] = {
            "html": artifact_bytes.decode("utf-8"),
            "revision": hashlib.sha256(artifact_bytes).hexdigest(),
            "path": declared_path,
        }
    return _cache[key]


def handle_custom_control(
    command: Mapping[str, Any], model: object, meta_path: Path
) -> dict[str, str] | None:
    """Return None unless the command is get-custom-control, so do_command can fall through."""
    if command.get("command") != CUSTOM_CONTROL_COMMAND:
        return None
    return load_custom_control(model, meta_path)


def clear_cache() -> None:
    _cache.clear()
