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

import m5
from m5.objects import *

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

cmd_bytes = 16
seu_base = 0x70000000
queue_depth = 2
poll_step_ticks = int(1e6)
max_poll_steps = 50000
expected_poll_exit_cause = "simulate() limit reached"
expected_completed_cmds = 5

system = System(
    mem_mode="timing",
    mem_ranges=[AddrRange("512MiB")],
    membus=SystemXBar(),
    physmem=SimpleMemory(range=AddrRange("512MiB")),
    clk_domain=SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain()),
)
system.system_port = system.membus.cpu_side_ports
system.physmem.port = system.membus.mem_side_ports

system.cpu = RiscvTimingSimpleCPU(cpu_id=0)
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.cpu.createInterruptController()

binary = os.path.abspath(args.binary)
system.workload = SEWorkload.init_compatible(binary)
process = Process(executable=binary)
process.cmd = [binary]
system.cpu.workload = process
system.cpu.createThreads()

system.seu = SpecializedExecutionUnit(
    base_addr=seu_base,
    macro_cmd_bytes=cmd_bytes,
    cmd_queue_depth=queue_depth,
    debug_process_latency="50ns",
)
system.seu.cpu_side = system.membus.mem_side_ports

root = Root(full_system=False, system=system)
m5.instantiate()

process.map(seu_base, seu_base, 2 * cmd_bytes, False)

all_done = False
exit_cause = ""
final_occupancy = 0
completed_cmds = 0
issue_busy = False
exit_cause_ok = True

for _ in range(max_poll_steps):
    exit_event = m5.simulate(poll_step_ticks)
    exit_cause = exit_event.getCause()

    if exit_cause != expected_poll_exit_cause:
        exit_cause_ok = False
        break

    final_occupancy = system.seu.queueOccupancy()
    completed_cmds = system.seu.completedCmdCount()
    issue_busy = system.seu.isIssueBusy()

    if completed_cmds >= expected_completed_cmds:
        all_done = True
        break

if all_done:
    print("SEU_EXIT_CAUSE=drain_complete")
else:
    print(f"SEU_EXIT_CAUSE={exit_cause}")
print(f"SEU_POLL_EXIT_OK={int(exit_cause_ok)}")
print(f"SEU_COMPLETED_CMDS={completed_cmds}")
print(f"SEU_FINAL_OCCUPANCY={final_occupancy}")
print(f"SEU_ISSUE_BUSY={issue_busy}")

if all_done and exit_cause_ok:
    print("SEU_TEST_PASS")
