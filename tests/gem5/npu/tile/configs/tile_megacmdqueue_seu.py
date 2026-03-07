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
seu_base = 0x71000000
queue_depth = 2
num_cmds = 6

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
)
system.cmdq.cpu_side = system.membus.mem_side_ports
system.cmdq.mem_side = system.membus.cpu_side_ports

system.seu = SpecializedExecutionUnit(
    base_addr=seu_base,
    macro_cmd_bytes=cmd_bytes,
    cmd_queue_depth=queue_depth,
    debug_process_latency="200ns",
    sync_enqueue_on_data_write=True,
)
system.seu.cpu_side = system.membus.mem_side_ports

root = Root(full_system=False, system=system)
m5.instantiate()

process.map(cmdq_base, cmdq_base, 2 * cmd_bytes, False)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
cmdq_occupancy = system.cmdq.queueOccupancy()
seu_occupancy = system.seu.queueOccupancy()
seu_completed = system.seu.completedCmdCount()
seu_busy = system.seu.isIssueBusy()

print(f"TILE_EXIT_CAUSE={exit_cause}")
print(f"TILE_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"TILE_SEU_OCCUPANCY={seu_occupancy}")
print(f"TILE_SEU_COMPLETED={seu_completed}")
print(f"TILE_SEU_BUSY={seu_busy}")

if (
    exit_cause == expected_exit_cause
    and cmdq_occupancy == 0
    and seu_occupancy == 0
    and not seu_busy
    and seu_completed >= num_cmds
):
    print("TEST_PASS")
