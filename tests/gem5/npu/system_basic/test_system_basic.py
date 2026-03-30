# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *

from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(re.compile(r"SYSTEM_BASIC_TEST_PASS"))

system_basic_dir = Path(__file__).resolve().parent
binary = system_basic_dir / "bin" / "system_basic_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(system_basic_dir / "src")),
)

gem5_verify_config(
    name="system_basic",
    verifiers=[pass_verifier],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "system_basic",
        "configs",
        "system_basic.py",
    ),
    config_args=["--binary", str(binary)],
    gem5_args=["--debug-flags=MegaCmdQueue,VPU"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
