#!/usr/bin/env bash
# Installs Node and pnpm when the build host lacks them, so the custom control UIs in this module
# can build on Viam's cloud builders. Reference it as build.setup in meta.json. It runs before
# build.build on every target platform and does nothing where the toolchain already exists.
set -euo pipefail

NODE_MAJOR=22
NODE_VERSION=22.23.1
PNPM_VERSION=10

log() {
  echo "setup.sh: $*" >&2
}

as_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

node_is_recent() {
  command -v node >/dev/null 2>&1 || return 1
  [ "$(node -p 'process.versions.node.split(".")[0]')" -ge "$NODE_MAJOR" ]
}

install_node_linux() {
  local arch
  case "$(uname -m)" in
    x86_64) arch=x64 ;;
    aarch64 | arm64) arch=arm64 ;;
    *)
      log "unsupported architecture $(uname -m)"
      exit 1
      ;;
  esac
  if ! command -v curl >/dev/null 2>&1; then
    as_root apt-get update
    as_root apt-get install -y --no-install-recommends curl ca-certificates
  fi
  local url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${arch}.tar.gz"
  log "installing Node ${NODE_VERSION} into /usr/local"
  curl -fsSL "$url" | as_root tar -xz -C /usr/local --strip-components=1
}

install_node_darwin() {
  log "installing Node with Homebrew"
  brew install node
}

install_pnpm() {
  local prefix
  prefix="$(npm prefix -g)"
  log "installing pnpm ${PNPM_VERSION}"
  if [ -w "$prefix/bin" ] && [ -w "$prefix/lib" ]; then
    npm install -g "pnpm@${PNPM_VERSION}"
  else
    as_root npm install -g "pnpm@${PNPM_VERSION}"
  fi
}

if ! node_is_recent; then
  case "$(uname -s)" in
    Linux) install_node_linux ;;
    Darwin) install_node_darwin ;;
    *)
      log "unsupported operating system $(uname -s)"
      exit 1
      ;;
  esac
fi

command -v pnpm >/dev/null 2>&1 || install_pnpm

log "node $(node --version), pnpm $(pnpm --version)"
