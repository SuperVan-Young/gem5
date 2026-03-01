# Copyright (c) 2026
# All rights reserved.

import re

from testlib import *

pass_verifier = verifier.MatchRegex(re.compile(r"MEGACMDQUEUE_TEST_PASS"))


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
    config_args=[],
    valid_isas=(constants.all_compiled_tag,),
)
