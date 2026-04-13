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

binary = resolve_binary_path(__file__, "mpu_basic_gemm_tiled_riscv")


register_npu_test(
    NpuRunnerSpec(
        name="mpu_basic_gemm_tiled",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(binary, "--scenario", "basic_gemm_tiled")
        ),
        gem5_args=(
            "--debug-flags=MpuUnit,MegaCmdQueue,"
            "ScratchpadMemory,DmaUnit",
        ),
        verifier_specs=(
            r"MPU_SUMMARY scenario=basic_gemm_tiled cmds=28 reads=24 "
            r"writes=8 .* mvin=8 load=8 compute=4 drain=4 mvout=4 .*",
            r"MPU_SCENARIO_PASS=basic_gemm_tiled",
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
