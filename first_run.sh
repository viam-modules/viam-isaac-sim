#!/usr/bin/env bash
# One-time machine setup, run automatically by viam-server when the module is
# first installed (meta.json "first_run").
#
# Turns a standard Ubuntu 22.04/24.04 x86_64 machine into one that can run
# Isaac Sim:
#   - system libraries kit needs (vulkan, GL)
#   - NVIDIA driver if none is present (a reboot may be needed after)
#   - Isaac Sim itself, uv-installed into a venv under the module data dir
#
# The isaacsim download is large (10GB+); if it exceeds viam-server's default
# first_run timeout, set "first_run_timeout": "2h0m0s" on the module config.
set -uo pipefail
# viam-server doesn't set a working directory for first_run; uv sync needs to
# run next to pyproject.toml
cd "$(dirname "$0")"

log() { echo "viam-isaac-sim first_run: $*"; }

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    log "not linux/x86_64 - nothing to install (mock mode still works)"
    exit 0
fi

cat /etc/os-release | grep VERSION= || echo "missing os-release file, skipping version log"


SUDO=""
if [ "$(id -u)" != "0" ]; then
   if sudo -n true 2>/dev/null; then
        SUDO="sudo -n"
    else
        log "WARNING: not root and no passwordless sudo; skipping apt steps"
    fi
fi

APT_DEPS="software-properties-common curl ca-certificates libvulkan1 vulkan-tools libglu1-mesa libegl1 libgomp1 libxt6 libxrandr2 libgl1 libglx0 libopengl0 libglvnd0"

# ---------------------------------------------------------------------------
# apt: system libs, python, gpu driver
# ---------------------------------------------------------------------------
if [ "$(id -u)" = "0" ] || [ -n "$SUDO" ]; then
    export DEBIAN_FRONTEND=noninteractive
    UPDATED=0
    if dpkg-query -s $APT_DEPS >& /dev/null; then
	    echo "all apt deps present: $APT_DEPS"
    else
	    echo "installing apt deps: $APT_DEPS"
	    $SUDO apt-get update -qq || log "WARNING: apt-get update failed; continuing"
	    UPDATED=1
	    $SUDO apt-get install -y -qq $APT_DEPS \
		|| log "WARNING: some system libraries failed to install"
    fi

    # Isaac Sim only supports validated driver branches; the R590/595 branch
    # is known to crash the RTX renderer (isaac-sim/IsaacSim#537, #643).
    # 580.x is the validated branch for Isaac 5.0 on Linux.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        if [ ${SKIP_NVIDIA_DRIVER:-0} = 1 ]; then
            echo "skipping nvidia driver install because SKIP_NVIDIA_DRIVER=1"
	else
            log "no NVIDIA driver found - installing the validated 580 branch"
            if [ $UPDATED = 0 ]; then
	        $SUDO apt-get update -qq || log "WARNING: apt-get update failed; continuing"
            fi
            $SUDO apt-get install -y -qq nvidia-driver-580 \
                || $SUDO apt-get install -y -qq nvidia-driver-580-open \
                || log "WARNING: driver install failed; install the 580-branch NVIDIA driver manually"
            log "NOTE: a REBOOT is likely required before the GPU is usable"
	fi
    else
        DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        log "NVIDIA driver present: ${DRIVER_VER:-unknown}"
        DRIVER_MAJOR="${DRIVER_VER%%.*}"
        case "$DRIVER_MAJOR" in
            ''|*[!0-9]*) ;;
            *)
                if [ "$DRIVER_MAJOR" -ge 590 ]; then
                    log "WARNING: driver $DRIVER_VER is NOT validated for isaac sim and is"
                    log "WARNING: known to crash the RTX renderer (librtx.scenedb crash) and"
                    log "WARNING: break CUDA init (cuDeviceGetUuid). Downgrade to the 580 branch:"
                    log "WARNING:   sudo apt-get install -y nvidia-driver-580 && sudo reboot"
                fi
                ;;
        esac
    fi
fi

export PATH="$PATH:$HOME/.local/bin"
if ! command -v uv >& /dev/null; then
	# todo: better way to manage min version of uv here
	curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ $VIAM_SIM_SANDBOX = 1 ]; then
	export UV_PROJECT_ENVIRONMENT=/opt/isaac-sim-venv
fi

# uv sync in first_run.sh, not run.sh, because it has a longer default timeout
export UV_NO_CACHE=false
uv cache dir
uv sync
