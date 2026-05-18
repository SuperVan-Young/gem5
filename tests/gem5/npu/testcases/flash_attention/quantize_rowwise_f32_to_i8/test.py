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

binary = resolve_binary_path(
    __file__, "flash_attention_quantize_rowwise_f32_to_i8_riscv"
)

register_npu_test(
    NpuRunnerSpec(
        name="flash_attention_quantize_rowwise_f32_to_i8",
        config=resolve_config_path(__file__),
        config_args=tuple(make_binary_config_args(binary)),
        verifier_specs=(
            r"FLASH_ATTENTION_QUANTIZE_ROWWISE_F32_TO_I8_PASS",
            r"FLASH_ATTENTION_QUANTIZE_ROWWISE_F32_TO_I8_CONFIG_PASS",
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
