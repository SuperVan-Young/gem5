# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *

from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(
    re.compile(r"SYSTEM_MULTIPORT_TEST_PASS")
)

system_multiport_dir = Path(__file__).resolve().parent
binary = system_multiport_dir / "bin" / "system_multiport_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(system_multiport_dir / "src")),
)

gem5_verify_config(
    name="system_multiport",
    verifiers=[pass_verifier],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "system_multiport",
        "configs",
        "system_multiport.py",
    ),
    config_args=["--binary", str(binary)],
    gem5_args=["--debug-flags=MegaCmdQueue,DMA,VPU,ScratchpadMemory"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
