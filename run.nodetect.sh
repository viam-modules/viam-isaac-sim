#!/usr/bin/env bash
# see Makefile for explanation of how + why this gets converted to run.sh
set -euo pipefail
# allow invocation from arbitrary working directory:
cd "$(dirname "$0")"

# pip-installed isaac sim prompts for the EULA on first boot and refuses to
# run as root without these
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-yes}
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export OMNI_KIT_ALLOW_ROOT=${OMNI_KIT_ALLOW_ROOT:-1}

export PATH="$PATH:$HOME/.local/bin"
if ! command -v uv >& /dev/null; then
	# todo: better way to manage min version of uv here
	curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ $VIAM_SIM_SANDBOX = 1 ]; then
	export UV_PROJECT_ENVIRONMENT=/opt/isaac-sim-venv
fi

exec uv run src/main.py "$@"
