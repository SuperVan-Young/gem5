# Copyright (c) 2026
# All rights reserved.

import argparse
import os

import m5
from m5.objects import *

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

cmd_width = 128
cmdq_base = 0x70000000
cmd_bytes = cmd_width // 8
expected_exit_cause = "exiting with last active thread context"
expected_final_occupancy = 2

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

system.cmdq = MegaCmdQueue(
    num_input_port=1,
    mega_cmd_width=cmd_width,
    cmd_queue_depth=2,
    base_addr=cmdq_base,
)
system.cmdq.cpu_side = system.membus.mem_side_ports

root = Root(full_system=False, system=system)
m5.instantiate()

process.map(cmdq_base, cmdq_base, 2 * cmd_bytes, False)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
final_occupancy = system.cmdq.queueOccupancy()

print(f"MEGACMDQUEUE_EXIT_CAUSE={exit_cause}")
print(f"MEGACMDQUEUE_FINAL_OCCUPANCY={final_occupancy}")

if (
    exit_cause == expected_exit_cause
    and final_occupancy == expected_final_occupancy
):
    print("MEGACMDQUEUE_TEST_PASS")
