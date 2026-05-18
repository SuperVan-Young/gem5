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

binary = resolve_binary_path(__file__, "flash_attention_basic_f32_riscv")

register_npu_test(
    NpuRunnerSpec(
        name="flash_attention_basic_f32",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(
                binary, "--scenario", "flash_attention_basic_f32"
            )
        ),
        gem5_args=(
            "--debug-flags=MpuUnit,MegaCmdQueue,ScratchpadMemory,DmaUnit",
        ),
        verifier_specs=(
            r"MPU_SUMMARY scenario=flash_attention_basic_f32 .*",
            r"FLASH_ATTENTION_BASIC_SCORES_I32=.*",
            r"FLASH_ATTENTION_BASIC_PROBS_F32=.*",
            r"FLASH_ATTENTION_BASIC_OUT_F32=.*",
            r"FLASH_ATTENTION_BASIC_F32_PASS",
            r"MPU_EXIT_CODE=0",
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
