# Copyright (c) 2026
# All rights reserved.

import re
from pathlib import Path

from testlib import *

from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(re.compile(r"VPU_DECODE_SMOKE_PASS"))

test_dir = Path(__file__).resolve().parent
binary = test_dir / "bin" / "vpu_decode_smoke_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(test_dir / "src")),
)

gem5_verify_config(
    name="vpu_decode_smoke",
    verifiers=[pass_verifier],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "vpu_decode_smoke",
        "configs",
        "vpu_decode_smoke.py",
    ),
    config_args=["--binary", str(binary)],
    gem5_args=["--debug-flags=VPU"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
