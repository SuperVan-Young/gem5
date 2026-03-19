# Copyright (c) 2026
# All rights reserved.

import argparse
import os

import m5
from m5.objects import *

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--scenario", required=True)
args = parser.parse_args()

cmd_width = 512
cmd_bytes = cmd_width // 8
cmdq_base = 0x70000000
dma_base = 0x74000000
dram_base = 0x20000000
spm_base = 0x60000000
spm_size = 64 * 1024
buffer_size = 32 if args.scenario == "buffer_size_forces_batching" else 4096
expected_exit_cause = "exiting with last active thread context"
expected_exit_code = 0

system = System(
    mem_mode="timing",
    mem_ranges=[
        AddrRange(0, size=0x60000000),
        AddrRange(spm_base, size=spm_size),
    ],
    membus=SystemXBar(),
    clk_domain=SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain()),
)
system.system_port = system.membus.cpu_side_ports

system.lowmem = SimpleMemory(range=AddrRange(0, size=0x60000000))
system.lowmem.port = system.membus.mem_side_ports

system.spm = ScratchpadMemory(
    range=AddrRange(spm_base, size=spm_size),
    latency="10ns",
    bandwidth="100GiB/s",
)
system.spm.port = system.membus.mem_side_ports

system.cpu = RiscvTimingSimpleCPU(cpu_id=0)
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.cpu.createInterruptController()

binary = os.path.abspath(args.binary)
system.workload = SEWorkload.init_compatible(binary)
process = Process(executable=binary)
process.cmd = [binary, args.scenario]
system.cpu.workload = process
system.cpu.createThreads()

system.cmdq = MegaCmdQueue(
    num_input_port=1,
    mega_cmd_width=cmd_width,
    cmd_queue_depth=8,
    base_addr=cmdq_base,
    range_addr=0x75000000,
    num_sync_indicator=256,
)
system.cmdq.cpu_side = system.membus.mem_side_ports
system.cmdq.sync_indicator_side = system.membus.mem_side_ports
system.cmdq.mem_side = system.membus.cpu_side_ports

system.dma = DmaUnit(
    base_addr=dma_base,
    macro_cmd_bytes=cmd_bytes,
    cmd_queue_depth=8,
    sync_enqueue_on_data_write=True,
    buffer_size=buffer_size,
)
system.dma.cpu_side = system.membus.mem_side_ports
system.dma.mem_side = system.membus.cpu_side_ports

root = Root(full_system=False, system=system)
m5.instantiate()

process.map(cmdq_base, cmdq_base, 2 * cmd_bytes, False)
process.map(dram_base, dram_base, 64 * 1024, False)
process.map(spm_base, spm_base, spm_size, False)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()

print(f"DMA_EXIT_CAUSE={exit_cause}")
print(f"DMA_EXIT_CODE={exit_code}")
print(f"DMA_SCENARIO={args.scenario}")

if (
    args.scenario != "invalid_address"
    and exit_cause == expected_exit_cause
    and exit_code == expected_exit_code
):
    print(f"DMA_SCENARIO_PASS={args.scenario}")
