# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *

from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(re.compile(r"SYSTEM_VPU_DUAL_TEST_PASS"))

system_vpu_dual_dir = Path(__file__).resolve().parent
binary = system_vpu_dual_dir / "bin" / "system_vpu_dual_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(system_vpu_dual_dir / "src")),
)

gem5_verify_config(
    name="system_vpu_dual",
    verifiers=[pass_verifier],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "system_vpu_dual",
        "configs",
        "system_vpu_dual.py",
    ),
    config_args=["--binary", str(binary)],
    gem5_args=["--debug-flags=MegaCmdQueue,VPU"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
