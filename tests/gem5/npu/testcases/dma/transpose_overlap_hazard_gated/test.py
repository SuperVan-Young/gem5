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

binary = resolve_binary_path(__file__, BINARY_NAME)

register_npu_test(
    NpuRunnerSpec(
        name=f"dma_{CASE_NAME}",
        config=resolve_config_path(__file__),
        config_args=tuple(make_binary_config_args(binary)),
        gem5_args=("--debug-flags=DmaUnit,MegaCmdQueue",),
        verifier_specs=(
            r"DMA_SUMMARY scenario=transpose_overlap_hazard_gated cmds=3 .* "
            r"overlap=0 .*",
            r"DMA_SCENARIO_PASS=transpose_overlap_hazard_gated",
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
