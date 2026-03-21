# Copyright (c) 2026
# All rights reserved.

import argparse
import os

import m5
from m5.objects import *

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--rounds", type=int, default=64)
args = parser.parse_args()

cmd_width = 128
cmd_bytes = cmd_width // 8
cmdq_base = 0x70000000
sync_base = 0x71000000
seu_base = 0x72000000
num_cpus = 4
num_input_port = 4
cmd_queue_depth = 8
expected_exit_cause = "exiting with last active thread context"

assert num_cpus == num_input_port
assert args.rounds > 0
expected_total = num_cpus * args.rounds

system = System(
    mem_mode="timing",
    mem_ranges=[AddrRange("512MiB")],
    membus=SystemXBar(),
    physmem=SimpleMemory(range=AddrRange("512MiB")),
    clk_domain=SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain()),
)
system.system_port = system.membus.cpu_side_ports
system.physmem.port = system.membus.mem_side_ports

cpus = [RiscvTimingSimpleCPU(cpu_id=i) for i in range(num_cpus)]
for cpu in cpus:
    cpu.icache_port = system.membus.cpu_side_ports
    cpu.dcache_port = system.membus.cpu_side_ports
    cpu.createInterruptController()

system.cpu = cpus

binary = os.path.abspath(args.binary)
system.workload = SEWorkload.init_compatible(binary)
processes = []

for cpu_id, cpu in enumerate(cpus):
    process = Process(executable=binary)
    process.pid = 100 + cpu_id
    process.cmd = [binary, str(cpu_id), str(args.rounds)]
    cpu.workload = process
    cpu.createThreads()
    processes.append(process)

system.cmdq = MegaCmdQueue(
    num_input_port=num_input_port,
    mega_cmd_width=cmd_width,
    cmd_queue_depth=cmd_queue_depth,
    base_addr=cmdq_base,
    range_addr=0x73000000,
    num_sync_indicator=256,
)
for _ in range(num_input_port):
    system.cmdq.cpu_side = system.membus.mem_side_ports
system.cmdq.sync_indicator_side = system.membus.mem_side_ports
system.cmdq.mem_side = system.membus.cpu_side_ports

system.seu = SpecializedExecutionUnit(
    base_addr=seu_base,
    macro_cmd_bytes=cmd_bytes,
    cmd_queue_depth=cmd_queue_depth,
    debug_process_latency="50ns",
    sync_enqueue_on_data_write=True,
)
system.seu.cpu_side = system.membus.mem_side_ports
system.seu.mem_side = system.membus.cpu_side_ports

root = Root(full_system=False, system=system)
m5.instantiate()

for cpu_id, process in enumerate(processes):
    port_base = cmdq_base + (cpu_id << 20)
    process.map(port_base, port_base, 2 * cmd_bytes, False)
    process.map(sync_base, sync_base, 4, False)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
last_cmdq_occupancy = system.cmdq.queueOccupancy()
last_seu_occupancy = system.seu.queueOccupancy()
last_seu_completed = system.seu.completedCmdCount()

print(f"FOUR_RV_EXIT_CAUSE={exit_cause}")
print(f"FOUR_RV_EXIT_OK={int(exit_cause == expected_exit_cause)}")
print(f"FOUR_RV_EXPECTED_TOTAL={expected_total}")
print(f"FOUR_RV_CMDQ_OCCUPANCY={last_cmdq_occupancy}")
print(f"FOUR_RV_SEU_OCCUPANCY={last_seu_occupancy}")
print(f"FOUR_RV_SEU_COMPLETED={last_seu_completed}")

if (
    exit_cause == expected_exit_cause
    and last_cmdq_occupancy == 0
    and last_seu_occupancy == 0
    and last_seu_completed == expected_total
):
    print("4RV_TEST_PASS")
