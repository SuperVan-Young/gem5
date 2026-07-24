#!/usr/bin/env bash

set -euo pipefail

SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SIM_DIR}/../.." && pwd)"

exec "${SIM_DIR}/gem5-fa-sim.sh" \
    --hardware "${REPO_DIR}/fproject/ppa_eval/configs/f7.yaml" \
    --task "${SIM_DIR}/configs/fa_q256_kv1024_d128.yaml" \
    --expect-utilization-min 40 \
    --expect-utilization-max 55 \
    "$@"
