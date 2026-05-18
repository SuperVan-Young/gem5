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

binary = resolve_binary_path(__file__, "flash_attention_qk_path_riscv")

register_npu_test(
    NpuRunnerSpec(
        name="flash_attention_qk_path",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(
                binary, "--scenario", "flash_attention_qk_path"
            )
        ),
        gem5_args=(
            "--debug-flags=MpuUnit,MegaCmdQueue,ScratchpadMemory,DmaUnit",
        ),
        verifier_specs=(
            r"MPU_SUMMARY scenario=flash_attention_qk_path cmds=28 "
            r"reads=22 writes=6 .* mvin=8 load=8 compute=4 drain=4 mvout=4 .*",
            r"FLASH_ATTENTION_QK_PATH_PASS",
            r"MPU_EXIT_CODE=0",
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
