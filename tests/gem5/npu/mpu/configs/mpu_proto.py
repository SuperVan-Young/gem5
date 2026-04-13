# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5
from m5.objects import AddrRange

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "configs"))

from npu_test_system import (
    NPUAddressMap,
    NPUTestSystemBuilder,
)

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--scenario", required=True)
args = parser.parse_args()

spm_size = 64 * 1024
if args.scenario == "spm_backpressure":
    spm_bandwidth = "1GiB/s"
    spm_latency = "20ns"
elif args.scenario == "prefetch_compute_overlap":
    spm_bandwidth = "1GiB/s"
    spm_latency = "100us"
else:
    spm_bandwidth = "100GiB/s"
    spm_latency = "10ns"
mem_ranges = [
    AddrRange(0, size=0x60000000),
    AddrRange(0x60000000, size=spm_size),
]

builder = NPUTestSystemBuilder(mem_ranges=mem_ranges, addr_map=NPUAddressMap())
builder.build_base_system()
builder.add_lowmem(AddrRange(0, size=0x60000000), attr_name="lowmem")
builder.add_spm(size=spm_size, latency=spm_latency, bandwidth=spm_bandwidth)
builder.add_cpu(cpu_id=0)
process = builder.set_workload(
    os.path.abspath(args.binary), argv=[args.scenario]
)
builder.add_megacmdqueue()
builder.add_mpu(attr_name="mpu", device_id=0, num_mem_side_ports=2)
if args.scenario == "multi_instance_route":
    builder.add_mpu(attr_name="mpu1", device_id=1, num_mem_side_ports=2)
if args.scenario == "dma_spm_mpu_chain":
    builder.add_dma(attr_name="dma", bank_size=4096)
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq(process=process)
builder.map_sync(process=process)
builder.map_spm(process=process)
builder.map_dram(process=process, size=64 * 1024)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()

active_attr = "mpu1" if args.scenario == "multi_instance_route" else "mpu"
active_mpu = getattr(builder.system, active_attr)

print(f"MPU_EXIT_CAUSE={exit_cause}")
print(f"MPU_EXIT_CODE={exit_code}")
print(f"MPU_SCENARIO={args.scenario}")
print(
    "MPU_SUMMARY "
    f"scenario={args.scenario} "
    f"cmds={active_mpu.completedCmdCount()} "
    f"reads={active_mpu.completedReadRespCount()} "
    f"writes={active_mpu.completedWriteRespCount()} "
    f"macro={active_mpu.macroFifoOccupancy()} "
    f"memq={active_mpu.memUopQueueOccupancy()} "
    f"execq={active_mpu.execUopQueueOccupancy()} "
    f"drainq={active_mpu.drainUopQueueOccupancy()} "
    f"mvin={active_mpu.mvinCmdCount()} "
    f"load={active_mpu.loadCmdCount()} "
    f"compute={active_mpu.computeCmdCount()} "
    f"drain={active_mpu.drainCmdCount()} "
    f"mvout={active_mpu.mvoutCmdCount()} "
    f"a0={active_mpu.aBufferState(0)} "
    f"a1={active_mpu.aBufferState(1)} "
    f"b0={active_mpu.bBufferState(0)} "
    f"b1={active_mpu.bBufferState(1)} "
    f"c0={active_mpu.cBufferState(0)} "
    f"c1={active_mpu.cBufferState(1)} "
    f"out={active_mpu.outputStorageStateCode()} "
    f"loadedA={active_mpu.loadedAIndex()} "
    f"loadedB={active_mpu.loadedBIndex()} "
    f"compute_cycles={active_mpu.lastComputeLatencyCycles()} "
    f"cmd_cycles={active_mpu.lastCommandLatencyCycles()} "
    f"spm_wait={active_mpu.stallCyclesWaitingForSpm()} "
    f"macs={active_mpu.totalMacOps()} "
    f"busy={active_mpu.busyCycles()} "
    f"idle={active_mpu.idleCycles()} "
    f"max_active_uops={active_mpu.maxActiveMicroOps()}"
)

if args.scenario == "multi_instance_route":
    print(
        "MPU_ROUTE_SUMMARY "
        f"mpu0_cmds={builder.system.mpu.completedCmdCount()} "
        f"mpu1_cmds={builder.system.mpu1.completedCmdCount()}"
    )

if args.scenario == "dma_spm_mpu_chain":
    print(
        "MPU_DMA_SUMMARY "
        f"dma_cmds={builder.system.dma.completedCmdCount()} "
        f"dma_reads={builder.system.dma.completedReadRespCount()} "
        f"dma_writes={builder.system.dma.completedWriteRespCount()}"
    )
