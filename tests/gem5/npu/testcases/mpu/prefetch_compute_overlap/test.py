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

binary = resolve_binary_path(__file__, "mpu_prefetch_compute_overlap_riscv")


register_npu_test(
    NpuRunnerSpec(
        name="mpu_prefetch_compute_overlap",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(
                binary, "--scenario", "prefetch_compute_overlap"
            )
        ),
        gem5_args=(
            "--debug-flags=MpuUnit,MegaCmdQueue,"
            "ScratchpadMemory,DmaUnit",
        ),
        verifier_specs=(
            r"MPU_SUMMARY scenario=prefetch_compute_overlap cmds=14 "
            r"reads=8 writes=4 .* mvin=4 load=4 compute=2 drain=2 "
            r"mvout=2 .* max_active_uops=[2-9][0-9]*",
            r"MPU_SCENARIO_PASS=prefetch_compute_overlap",
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
