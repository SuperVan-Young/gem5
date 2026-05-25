# Copyright (c) 2026
# All rights reserved.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from runner_common import (  # noqa: E402
    NpuRunnerSpec,
    NpuScenarioSpec,
    make_binary_config_args,
    make_testcase_build_fixture,
    register_npu_scenarios,
    resolve_binary_path,
    resolve_config_path,
)

binary = resolve_binary_path(__file__, "vpu_unary_exp_riscv")

register_npu_scenarios(
    NpuRunnerSpec(
        name="vpu_unary_exp",
        config=resolve_config_path(__file__),
        config_args=tuple(make_binary_config_args(binary)),
        gem5_args=("--debug-flags=VPU",),
        verifier_specs=r"VPU_UNARY_EXP_PASS",
        fixtures=(make_testcase_build_fixture(__file__),),
    ),
    (
        NpuScenarioSpec(suffix="dlen4"),
        NpuScenarioSpec(
            suffix="dlen16",
            config_args=("--dlen-bytes", "16"),
        ),
    ),
)
