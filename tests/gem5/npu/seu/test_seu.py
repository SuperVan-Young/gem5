# Copyright (c) 2026
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
from pathlib import Path

from testlib import *

from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)

pass_verifier = verifier.MatchRegex(re.compile(r"SEU_TEST_PASS"))

seu_dir = Path(__file__).resolve().parent
binary = seu_dir / "bin" / "seu_mmio_riscv"
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(seu_dir / "src")),
)

gem5_verify_config(
    name="seu_basic",
    verifiers=[pass_verifier],
    config=joinpath(
        config.base_dir,
        "tests",
        "gem5",
        "npu",
        "seu",
        "configs",
        "seu_basic.py",
    ),
    config_args=["--binary", str(binary)],
    gem5_args=["--debug-flags=SpecializedExecutionUnit"],
    valid_isas=(constants.riscv_tag,),
    valid_hosts=constants.supported_hosts,
    length=constants.quick_tag,
    fixtures=[build_fixture],
)
