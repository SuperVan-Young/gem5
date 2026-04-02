# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5
from m5.objects import AddrRange

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--scenario", required=True)
args = parser.parse_args()

config_dir = Path(__file__).resolve().parent
sys.path.append(str(config_dir.parents[1] / "configs"))

from npu_test_system import NPUTestSystemBuilder

spm_base = 0x60000000
spm_size = 64 * 1024
dram_size = 256 * 1024
expected_exit_cause = "exiting with last active thread context"
expected_exit_code = 0
needs_dma = args.scenario == "dma_chain"

builder = NPUTestSystemBuilder(
    mem_ranges=[
        AddrRange(0, size=0x60000000),
        AddrRange(spm_base, size=spm_size),
    ]
)
builder.build_base_system()
builder.add_lowmem(AddrRange(0, size=0x60000000))
builder.add_spm(base_addr=spm_base, size=spm_size)
builder.add_cpu(cpu_id=0)
builder.set_workload(os.path.abspath(args.binary), [args.scenario], cpu_id=0)
builder.add_megacmdqueue()
builder.add_mpu(
    load_bandwidth_bytes_per_cycle=8,
    store_bandwidth_bytes_per_cycle=8,
    local_bank_count=2,
    local_bank_granularity_bytes=4,
    local_bank_service_cycles=1,
)
if needs_dma:
    builder.add_dma()
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq()
builder.map_spm()
builder.map_dram(size=dram_size)
builder.map_mpu()

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()

print(f"MPU_EXIT_CAUSE={exit_cause}")
print(f"MPU_EXIT_CODE={exit_code}")
print(f"MPU_SCENARIO={args.scenario}")
print(
    "MPU_SUMMARY "
    f"scenario={args.scenario} "
    f"cmds={builder.system.mpu.completedCmdCount()} "
    f"loads={builder.system.mpu.loadCmdCount()} "
    f"computes={builder.system.mpu.computeCmdCount()} "
    f"stores={builder.system.mpu.storeCmdCount()} "
    f"loops={builder.system.mpu.tensorLoopCmdCount()} "
    f"matmul={builder.system.mpu.matmulCount()} "
    f"matmul_acc={builder.system.mpu.matmulAccCount()} "
    f"reads={builder.system.mpu.completedReadRespCount()} "
    f"writes={builder.system.mpu.completedWriteRespCount()} "
    f"iters={builder.system.mpu.completedIterationCount()} "
    f"queue={builder.system.mpu.queueOccupancy()} "
    f"cmdq={builder.system.cmdq.queueOccupancy()} "
    f"busy={int(builder.system.mpu.isIssueBusy())} "
    f"active={builder.system.mpu.maxActiveMicroOps()} "
    f"tiles={builder.system.mpu.tensorLoopExpandedTiles()} "
    f"acc_tiles={builder.system.mpu.tensorLoopExpandedAccTiles()} "
    f"internal_loads={builder.system.mpu.totalInternalLoads()} "
    f"internal_computes={builder.system.mpu.totalInternalComputes()} "
    f"internal_stores={builder.system.mpu.totalInternalStores()} "
    f"total_tiles={builder.system.mpu.totalTiles()} "
    f"total_acc_tiles={builder.system.mpu.totalAccTiles()} "
    f"spm_stall={builder.system.mpu.stallCyclesWaitingForSPM()} "
    f"slot_stall={builder.system.mpu.stallCyclesWaitingForSlot()} "
    f"latency={builder.system.mpu.computedTotalLatency()} "
    f"a0={int(builder.system.mpu.slotA0Valid())} "
    f"a1={int(builder.system.mpu.slotA1Valid())} "
    f"b0={int(builder.system.mpu.slotB0Valid())} "
    f"b1={int(builder.system.mpu.slotB1Valid())} "
    f"c0={int(builder.system.mpu.slotC0Valid())} "
    f"c1={int(builder.system.mpu.slotC1Valid())} "
    f"c0dirty={int(builder.system.mpu.slotC0Dirty())} "
    f"c1dirty={int(builder.system.mpu.slotC1Dirty())}"
)

if needs_dma:
    print(
        "DMA_SUMMARY "
        f"scenario={args.scenario} "
        f"cmds={builder.system.dma.completedCmdCount()} "
        f"reads={builder.system.dma.completedReadRespCount()} "
        f"writes={builder.system.dma.completedWriteRespCount()} "
        f"iters={builder.system.dma.completedIterationCount()} "
        f"queue={builder.system.dma.queueOccupancy()} "
        f"busy={int(builder.system.dma.isIssueBusy())} "
        f"active={builder.system.dma.maxActiveMicroOps()}"
    )

if exit_cause == expected_exit_cause and exit_code == expected_exit_code:
    print(f"MPU_SCENARIO_PASS={args.scenario}")
