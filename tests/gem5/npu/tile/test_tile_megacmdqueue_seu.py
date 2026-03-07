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

tile_dir = Path(__file__).resolve().parent
binary = tile_dir / "bin" / "tile_megacmdqueue_seu_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(tile_dir / "src")),
)

gem5_verify_config(
    name="tile_megacmdqueue_seu",
    verifiers=[pass_verifier],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "tile",
        "configs",
        "tile_megacmdqueue_seu.py",
    ),
    config_args=["--binary", str(binary)],
    gem5_args=["--debug-flags=MegaCmdQueue,SpecializedExecutionUnit"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
