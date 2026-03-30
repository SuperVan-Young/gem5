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
# neither the name of the copyright holders nor the names of their
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

expected_exit_cause = "exiting with last active thread context"
expected_completed_cmds = 1
read_mask = 0x0000000B  # ports 0, 1, 3
write_mask = 0x0000000E  # ports 1, 2, 3
repetition = 3


def popcount32(value):
    count = 0
    while value:
        count += value & 1
        value >>= 1
    return count


expected_prologues = repetition
expected_executes = repetition
expected_epilogues = repetition
expected_read_resps = popcount32(read_mask) * repetition
expected_write_resps = popcount32(write_mask) * repetition
expected_iterations = repetition
expected_min_overlap = 2

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_spm()
builder.add_cpu(cpu_id=0)
builder.set_workload(binary, cpu_id=0)
builder.add_megacmdqueue()
builder.add_seu(num_mem_side_ports=4)
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq()
builder.map_spm()
builder.map_seu()

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
completed_cmds = builder.system.seu.completedCmdCount()
prologues = builder.system.seu.prologueCount()
read_resps = builder.system.seu.completedReadRespCount()
executes = builder.system.seu.executeCount()
write_resps = builder.system.seu.completedWriteRespCount()
epilogues = builder.system.seu.epilogueCount()
iterations = builder.system.seu.completedIterationCount()
max_active_micro_ops = builder.system.seu.maxActiveMicroOps()

print(f"SEU_MULTI_EXIT_CAUSE={exit_cause}")
print(f"SEU_MULTI_COMPLETED_CMDS={completed_cmds}")
print(f"SEU_MULTI_PROLOGUES={prologues}")
print(f"SEU_MULTI_READ_RESPS={read_resps}")
print(f"SEU_MULTI_EXECUTES={executes}")
print(f"SEU_MULTI_WRITE_RESPS={write_resps}")
print(f"SEU_MULTI_EPILOGUES={epilogues}")
print(f"SEU_MULTI_ITERATIONS={iterations}")
print(f"SEU_MULTI_MAX_ACTIVE_MICRO_OPS={max_active_micro_ops}")

if (
    exit_cause == expected_exit_cause
    and completed_cmds == expected_completed_cmds
    and prologues == expected_prologues
    and read_resps == expected_read_resps
    and executes == expected_executes
    and write_resps == expected_write_resps
    and epilogues == expected_epilogues
    and iterations == expected_iterations
    and max_active_micro_ops >= expected_min_overlap
):
    print("SEU_MULTI_TEST_PASS")
