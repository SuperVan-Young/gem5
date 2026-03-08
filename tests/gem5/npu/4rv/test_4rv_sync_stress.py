# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *

from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(re.compile(r"4RV_TEST_PASS"))

four_rv_dir = Path(__file__).resolve().parent
binary = four_rv_dir / "bin" / "4rv_sync_stress_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(four_rv_dir / "src")),
)

gem5_verify_config(
    name="4rv_sync_stress",
    verifiers=[pass_verifier],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "4rv",
        "configs",
        "4rv_sync_stress.py",
    ),
    config_args=["--binary", str(binary), "--rounds", "32"],
    gem5_args=["--debug-flags=MegaCmdQueue,SpecializedExecutionUnit"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
