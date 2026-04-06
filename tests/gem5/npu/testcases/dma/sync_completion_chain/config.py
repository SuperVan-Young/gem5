# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5
from m5.objects import *

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import verify_snapshot  # noqa: E402

SCENARIO_NAME = "sync_completion"
EXPECTED_SNAPSHOT = {
    "exit_cause": "exiting with last active thread context",
    "exit_code": 0,
    "scenario": SCENARIO_NAME,
    "cmds": 2,
    "queue": 0,
    "cmdq": 0,
    "busy": 0,
}

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

cmd_width = 512
cmd_bytes = cmd_width // 8
cmdq_base = 0x70000000
dma_base = 0x74000000
dram_base = 0x20000000
spm_base = 0x60000000
spm_size = 64 * 1024
bank_size = 4096
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
process.cmd = [binary, SCENARIO_NAME]
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

dma_mem_ports = 2
system.dma = DmaUnit(
    base_addr=dma_base,
    macro_cmd_bytes=cmd_bytes,
    cmd_queue_depth=8,
    sync_enqueue_on_data_write=True,
    bank_size=bank_size,
    num_mem_side_ports=dma_mem_ports,
)
system.dma.cpu_side = system.membus.mem_side_ports
for _ in range(dma_mem_ports):
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
print(f"DMA_SCENARIO={SCENARIO_NAME}")

snapshot = {
    "exit_cause": exit_cause,
    "exit_code": exit_code,
    "scenario": SCENARIO_NAME,
    "cmds": system.dma.completedCmdCount(),
    "reads": system.dma.completedReadRespCount(),
    "writes": system.dma.completedWriteRespCount(),
    "iters": system.dma.completedIterationCount(),
    "queue": system.dma.queueOccupancy(),
    "cmdq": system.cmdq.queueOccupancy(),
    "busy": int(system.dma.isIssueBusy()),
    "active": system.dma.maxActiveMicroOps(),
}

print(
    "DMA_SUMMARY "
    f"scenario={snapshot['scenario']} "
    f"cmds={snapshot['cmds']} "
    f"reads={snapshot['reads']} "
    f"writes={snapshot['writes']} "
    f"iters={snapshot['iters']} "
    f"queue={snapshot['queue']} "
    f"cmdq={snapshot['cmdq']} "
    f"busy={snapshot['busy']} "
    f"active={snapshot['active']}"
)

if verify_snapshot(
    {key: snapshot[key] for key in EXPECTED_SNAPSHOT},
    EXPECTED_SNAPSHOT,
):
    print(f"DMA_SCENARIO_PASS={SCENARIO_NAME}")
