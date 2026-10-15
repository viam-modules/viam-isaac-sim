#!/usr/bin/env bash
# run 3 seeded picks

set -euo pipefail

export VIAM_API_KEY="$(jq -r '.cloud.api_key.key' /etc/viam.json)"
export VIAM_API_KEY_ID="$(jq -r '.cloud.api_key.id' /etc/viam.json)"

for seed in 54 55 56; do
  echo "=== seed $seed ==="
  .venv/bin/python examples/scripted_pick.py localhost:8080 \
    --disable-webrtc \
    --seed "$seed" \
    --prop pick_block \
    --grasp-height-m 0.0125 \
    || echo "seed $seed failed"
done
