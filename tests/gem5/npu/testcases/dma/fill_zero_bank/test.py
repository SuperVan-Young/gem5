# Copyright (c) 2026
# All rights reserved.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from runner_common import (  # noqa: E402
    NpuRunnerSpec,
    make_binary_config_args,
    make_testcase_build_fixture,
    register_npu_test,
    resolve_binary_path,
    resolve_config_path,
)

CASE_NAME = Path(__file__).resolve().parent.name
BINARY_NAME = f"dma_{CASE_NAME}_riscv"
SCENARIO_NAME_MAP = {
    "fill_zero_bank": "fill_zero_bank",
    "fill_zero_dram": "fill_zero_dram",
    "fill_zero_spm": "fill_zero_spm",
    "fill_nonzero_bank": "fill_nonzero_bank",
    "fill_nonzero_dram": "fill_nonzero_dram",
    "fill_nonzero_spm": "fill_nonzero_spm",
    "transpose_hw": "transpose_hw",
    "transpose_hc": "transpose_hc",
    "transpose_wc": "transpose_wc",
    "sync_completion_chain": "sync_completion",
    "queued_chain_basic": "queued_chain",
}
SCENARIO_NAME = SCENARIO_NAME_MAP[CASE_NAME]

TRANSPOSE_HW_LATENCY_REGEX = (
    r".*DMA_TRANSPOSE_LATENCY dim_a=0 dim_b=1 extent_a=2 extent_b=4 "
    r"extent_rest=3 transpose_unit_latency=1000 "
    r"computed_total_latency=24000"
)
TRANSPOSE_HW_SUMMARY_REGEX = (
    r"DMA_SUMMARY scenario=transpose_hw cmds=1 .* iters=3 .* "
    r"active=[2-9][0-9]*"
)
TRANSPOSE_HC_LATENCY_REGEX = (
    r".*DMA_TRANSPOSE_LATENCY dim_a=0 dim_b=2 extent_a=2 extent_b=4 "
    r"extent_rest=3 transpose_unit_latency=1000 "
    r"computed_total_latency=24000"
)
TRANSPOSE_WC_LATENCY_REGEX = (
    r".*DMA_TRANSPOSE_LATENCY dim_a=1 dim_b=2 extent_a=2 extent_b=4 "
    r"extent_rest=3 transpose_unit_latency=1000 "
    r"computed_total_latency=24000"
)

VERIFIER_SPECS = {
    "fill_zero_bank": (
        r".*DMA_BANK_FILL_OBSERVE bank=1 value=0 required=64 checksum=0",
        r"DMA_SUMMARY scenario=fill_zero_bank cmds=1 reads=0 writes=0 .*",
        r"DMA_SCENARIO_PASS=fill_zero_bank",
    ),
    "fill_zero_dram": (
        r"DMA_SUMMARY scenario=fill_zero_dram cmds=1 reads=0 writes=1 .*",
        r"DMA_SCENARIO_PASS=fill_zero_dram",
    ),
    "fill_zero_spm": (
        r"DMA_SUMMARY scenario=fill_zero_spm cmds=1 reads=0 writes=1 .*",
        r"DMA_SCENARIO_PASS=fill_zero_spm",
    ),
    "fill_nonzero_bank": (
        r".*DMA_BANK_FILL_OBSERVE bank=1 value=90 required=64 checksum=5760",
        r"DMA_SUMMARY scenario=fill_nonzero_bank cmds=1 reads=0 writes=0 .*",
        r"DMA_SCENARIO_PASS=fill_nonzero_bank",
    ),
    "fill_nonzero_dram": (
        r"DMA_SUMMARY scenario=fill_nonzero_dram cmds=1 reads=0 writes=1 .*",
        r"DMA_SCENARIO_PASS=fill_nonzero_dram",
    ),
    "fill_nonzero_spm": (
        r"DMA_SUMMARY scenario=fill_nonzero_spm cmds=1 reads=0 writes=1 .*",
        r"DMA_SCENARIO_PASS=fill_nonzero_spm",
    ),
    "transpose_hw": (
        TRANSPOSE_HW_LATENCY_REGEX,
        TRANSPOSE_HW_SUMMARY_REGEX,
        r"DMA_SCENARIO_PASS=transpose_hw",
    ),
    "transpose_hc": (
        TRANSPOSE_HC_LATENCY_REGEX,
        r"DMA_SCENARIO_PASS=transpose_hc",
    ),
    "transpose_wc": (
        TRANSPOSE_WC_LATENCY_REGEX,
        r"DMA_SCENARIO_PASS=transpose_wc",
    ),
    "sync_completion_chain": (r"DMA_SCENARIO_PASS=sync_completion",),
    "queued_chain_basic": (
        r"DMA_SUMMARY scenario=queued_chain cmds=2 .* queue=0 cmdq=0 busy=0",
        r"DMA_SCENARIO_PASS=queued_chain",
    ),
}

binary = resolve_binary_path(__file__, BINARY_NAME)

register_npu_test(
    NpuRunnerSpec(
        name=f"dma_{CASE_NAME}",
        config=resolve_config_path(__file__),
        config_args=tuple(make_binary_config_args(binary)),
        gem5_args=("--debug-flags=DmaUnit,MegaCmdQueue",),
        verifier_specs=VERIFIER_SPECS[CASE_NAME],
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
