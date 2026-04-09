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

binary = resolve_binary_path(__file__, "mpu_output_stationary_basic_riscv")


register_npu_test(
    NpuRunnerSpec(
        name="mpu_output_stationary_basic",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(
                binary, "--scenario", "output_stationary_basic"
            )
        ),
        gem5_args=(
            "--debug-flags=MpuUnit,MegaCmdQueue,"
            "ScratchpadMemory,DmaUnit",
        ),
        verifier_specs=(
            r"MPU_SUMMARY scenario=output_stationary_basic cmds=7 .* "
            r"mvin=2 load=2 compute=1 drain=1 mvout=1 .*",
            r"MPU_SCENARIO_PASS=output_stationary_basic",
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
