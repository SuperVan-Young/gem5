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

binary = resolve_binary_path(__file__, "swiglu_f32_riscv")

register_npu_test(
    NpuRunnerSpec(
        name="swiglu_f32",
        config=resolve_config_path(__file__),
        config_args=tuple(make_binary_config_args(binary)),
        gem5_args=("--debug-flags=VPU",),
        verifier_specs=r"SWIGLU_F32_PASS",
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
