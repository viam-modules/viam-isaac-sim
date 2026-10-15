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

. env.sh

exec uv run --extra full src/main.py "$@"
