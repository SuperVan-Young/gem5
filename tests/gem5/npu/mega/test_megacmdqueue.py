# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *
from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(re.compile(r"MEGACMDQUEUE_TEST_PASS"))

mega_dir = Path(__file__).resolve().parent
binary = mega_dir / "bin" / "megacmdqueue_mmio_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(mega_dir / "src")),
)

gem5_verify_config(
    name="megacmdqueue_full",
    verifiers=[pass_verifier],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "mega",
        "configs",
        "megacmdqueue_full.py",
    ),
    config_args=["--binary", str(binary)],
    gem5_args=["--debug-flags=MegaCmdQueue"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
