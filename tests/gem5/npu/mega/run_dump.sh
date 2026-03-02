#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gem5"
OUT_DIR="tests/gem5/npu/mega/run_dump"
SIM_OUT_DIR="${OUT_DIR}/simoutdir"
BIN_REL="tests/gem5/npu/mega/bin/megacmdqueue_mmio_riscv"
CFG_REL="tests/gem5/npu/mega/configs/megacmdqueue_full.py"

# 1) Build the RISCV test binary
cd "${REPO_ROOT}/tests/gem5/npu/mega/src"
make clean
make

# 2) Prepare output directory
cd "${REPO_ROOT}"
rm -rf "${SIM_OUT_DIR}"
mkdir -p "${SIM_OUT_DIR}"

# 3) Run gem5 with debug flags and dump logs under tests/
build/RISCV/gem5.opt \
  -d "${SIM_OUT_DIR}" \
  --debug-flags=MegaCmdQueue \
  "${CFG_REL}" \
  --binary "${BIN_REL}" \
  > "${OUT_DIR}/run.stdout" \
  2> "${OUT_DIR}/run.stderr"

echo "Run complete."
echo "  stdout: ${REPO_ROOT}/${OUT_DIR}/run.stdout"
echo "  stderr: ${REPO_ROOT}/${OUT_DIR}/run.stderr"
