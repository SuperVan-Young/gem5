# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *
from testlib.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(re.compile(r"MEGACMDQUEUE_TEST_PASS"))
push_accept_verifier = verifier.MatchRegex(re.compile(r"push accepted"))
push_reject_verifier = verifier.MatchRegex(re.compile(r"push rejected"))
pop_verifier = verifier.MatchRegex(re.compile(r"pop executed"))
control_push_verifier = verifier.MatchRegex(
    re.compile(r"control write addr=.*val=0")
)
control_pop_verifier = verifier.MatchRegex(
    re.compile(r"control write addr=.*val=1")
)

mega_dir = Path(__file__).resolve().parent
binary = mega_dir / "bin" / "megacmdqueue_mmio_riscv"

build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(mega_dir / "src")),
)

gem5_verify_config(
    name="megacmdqueue_full",
    verifiers=[
        pass_verifier,
        push_accept_verifier,
        push_reject_verifier,
        pop_verifier,
        control_push_verifier,
        control_pop_verifier,
    ],
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
    fixtures=[build_fixture],
)
