#!/usr/bin/env bash
# see Makefile for explanation of how + why this gets converted to run.sh
set -euo pipefail
# allow invocation from arbitrary working directory:
cd "$(dirname "$0")"

export PATH="$PATH:$HOME/.local/bin"
if ! command -v uv >& /dev/null; then
	# todo: better way to manage min version of uv here
	curl -LsSf https://astral.sh/uv/install.sh | sh
fi

exec uv run src/main.py "$@"
