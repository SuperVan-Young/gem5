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
cmd_bytes = cmd_width // 8
cmdq_base = 0x70000000
seu_base = 0x72000000
queue_depth = 4
expected_exit_cause = "exiting with last active thread context"

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
    cmd_queue_depth=queue_depth,
    base_addr=cmdq_base,
    range_addr=0x73000000,
    num_sync_indicator=256,
)
system.cmdq.cpu_side = system.membus.mem_side_ports
system.cmdq.sync_indicator_side = system.membus.mem_side_ports
system.cmdq.mem_side = system.membus.cpu_side_ports

system.seu = SpecializedExecutionUnit(
    base_addr=seu_base,
    macro_cmd_bytes=cmd_bytes,
    cmd_queue_depth=queue_depth,
    debug_process_latency="50ns",
    sync_enqueue_on_data_write=True,
)
system.seu.cpu_side = system.membus.mem_side_ports
system.seu.mem_side = system.membus.cpu_side_ports

root = Root(full_system=False, system=system)
m5.instantiate()

process.map(cmdq_base, cmdq_base, 2 * cmd_bytes, False)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
cmdq_occupancy = system.cmdq.queueOccupancy()
seu_occupancy = system.seu.queueOccupancy()
seu_completed = system.seu.completedCmdCount()

print(f"SIT_EXIT_CAUSE={exit_cause}")
print(f"SIT_EXIT_OK={int(exit_cause == expected_exit_cause)}")
print(f"SIT_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"SIT_SEU_OCCUPANCY={seu_occupancy}")
print(f"SIT_SEU_COMPLETED={seu_completed}")

if (
    exit_cause == expected_exit_cause
    and cmdq_occupancy == 0
    and seu_occupancy == 0
    and seu_completed == 2
):
    print("TEST_PASS")
