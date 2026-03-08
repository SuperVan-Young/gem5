# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *

from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(re.compile(r"TEST_PASS"))

sit_dir = Path(__file__).resolve().parent
binary = sit_dir / "bin" / "sit_launch_sync_launch_sync_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(sit_dir / "src")),
)

gem5_verify_config(
    name="sit_launch_sync_launch_sync",
    verifiers=[
        pass_verifier,
    ],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "sit",
        "configs",
        "sit_launch_sync_launch_sync.py",
    ),
    config_args=["--binary", str(binary)],
    gem5_args=["--debug-flags=MegaCmdQueue,SpecializedExecutionUnit"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
