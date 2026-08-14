#!/usr/bin/env bash

set -euo pipefail

SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SIM_DIR}/../../.." && pwd)"
CONTAINER_NAME="${USER}.gem5"
CONTAINER_ARGS=()

for arg in "$@"; do
    if [[ "${arg}" == "${REPO_DIR}"/* ]]; then
        CONTAINER_ARGS+=("/gem5/${arg#"${REPO_DIR}/"}")
    else
        CONTAINER_ARGS+=("${arg}")
    fi
done

exec docker exec -i "${CONTAINER_NAME}" bash -lc \
    'cd /gem5 && exec python3 fproject-atlas/sim/softmax/run.py "$@"' \
    atlas-softmax "${CONTAINER_ARGS[@]}"
