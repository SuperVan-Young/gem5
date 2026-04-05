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

binary = resolve_binary_path(__file__, "4rv_sync_stress_riscv")


register_npu_test(
    NpuRunnerSpec(
        name="megacmdqueue_4rv_sync_stress",
        config=resolve_config_path(__file__),
        config_args=tuple(make_binary_config_args(binary, "--rounds", 32)),
        gem5_args=("--debug-flags=MegaCmdQueue,SpecializedExecutionUnit",),
        verifier_specs=r"MEGACMDQUEUE_4RV_SYNC_STRESS_PASS",
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
