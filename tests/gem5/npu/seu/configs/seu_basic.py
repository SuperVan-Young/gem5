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

import argparse
import os
import sys
from pathlib import Path

import m5

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

sys.path.append(str(Path(__file__).resolve().parents[2] / "configs"))

from npu_test_system import NPUTestSystemBuilder

expected_completed_cmds = 5
expected_exit_cause = "exiting with last active thread context"

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_cpu(cpu_id=0)
builder.set_workload(binary, cpu_id=0)
builder.add_megacmdqueue()
builder.add_seu()
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq()
builder.map_seu()

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
final_occupancy = builder.system.seu.queueOccupancy()
completed_cmds = builder.system.seu.completedCmdCount()
issue_busy = builder.system.seu.isIssueBusy()

print(f"SEU_EXIT_CAUSE={exit_cause}")
print(f"SEU_COMPLETED_CMDS={completed_cmds}")
print(f"SEU_FINAL_OCCUPANCY={final_occupancy}")
print(f"SEU_ISSUE_BUSY={issue_busy}")

if (
    exit_cause == expected_exit_cause
    and completed_cmds == expected_completed_cmds
    and final_occupancy == 0
    and not issue_busy
):
    print("SEU_TEST_PASS")
