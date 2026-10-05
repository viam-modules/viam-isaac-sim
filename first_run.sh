#!/usr/bin/env bash
# One-time setup, run automatically by viam-server when the module is first
# installed (meta.json "first_run"). The module itself is plain python; Isaac
# Sim runs separately with the viam_isaac_server extension (see README).
set -euo pipefail
# viam-server doesn't set a working directory for first_run; uv sync needs to
# run next to pyproject.toml
cd "$(dirname "$0")"

export PATH="$PATH:$HOME/.local/bin"
if ! command -v uv >& /dev/null; then
	# todo: better way to manage min version of uv here
	curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv sync in first_run.sh, not run.sh, because it has a longer default timeout
export UV_NO_CACHE=false
uv cache dir
uv sync
