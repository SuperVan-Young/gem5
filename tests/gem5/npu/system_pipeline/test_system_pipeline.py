# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *

from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

system_pipeline_dir = Path(__file__).resolve().parent
binary = system_pipeline_dir / "bin" / "system_pipeline_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(system_pipeline_dir / "src")),
)

config_path = joinpath(
    config.base_dir,
    "tests",
    "gem5",
    "npu",
    "system_pipeline",
    "configs",
    "system_pipeline.py",
)


def add_system_pipeline_test(name, scenario):
    gem5_verify_config(
        name=name,
        verifiers=[
            verifier.MatchRegex(
                re.compile(
                    rf"SYSTEM_PIPELINE_TEST_PASS={re.escape(scenario)}"
                )
            )
        ],
        config=config_path,
        config_args=["--binary", str(binary), "--scenario", scenario],
        gem5_args=["--debug-flags=MegaCmdQueue,DMA,VPU,ScratchpadMemory"],
        valid_isas=(constants.riscv_tag,),
        valid_hosts=constants.supported_hosts,
        length=constants.quick_tag,
        fixtures=[build_fixture],
    )


add_system_pipeline_test("system_pipeline_spm_only", "spm_only")
add_system_pipeline_test("system_pipeline_copy_back", "copy_back")
