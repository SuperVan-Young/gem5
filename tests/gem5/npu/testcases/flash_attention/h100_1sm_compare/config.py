# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import os
import sys
from pathlib import Path

import m5
from m5.objects import AddrRange

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_system import (  # noqa: E402
    NPUAddressMap,
    NPUTestSystemBuilder,
)

SCENARIO = "flash_attention_h100_1sm_compare"
SYSTEM_CLOCK = "1GHz"
SPM_SIZE_BYTES = 256 * 1024
SPM_LATENCY = "10ns"
SPM_BANDWIDTH = "100GiB/s"
DRAM_MAP_SIZE_BYTES = 32 * 1024 * 1024
DRAM_BANDWIDTH = "25.378787879GB/s"
DRAM_LATENCY = "30ns"

# This square array is only a tile-scale approximation for the tensor path.
MPU_ARRAY_DIM = 128
MPU_A_BUFFER_BYTES = 16 * 1024
MPU_B_BUFFER_BYTES = 16 * 1024
MPU_C_BUFFER_BYTES = 64 * 1024
MPU_MEM_SIDE_PORTS = 2
PROFILE_LOG = (
    Path(__file__).resolve().parent
    / "profile"
    / "h100_1sm_compare.npu_profile.log"
)
TICKS_PER_CYCLE = 1000
PROBE_SYNC_INDICATOR = 0x11
COMPARE_QK_SYNC_INDICATOR = 0x21
COMPARE_PV_SYNC_INDICATOR = 0x31


def _parse_profile_end_events(profile_log):
    events = []
    for line in profile_log.read_text(encoding="utf-8").splitlines():
        if "NPU_PROFILE " not in line:
            continue
        payload = json.loads(line.split("NPU_PROFILE ", 1)[1])
        if payload.get("event") != "end":
            continue
        duration = int(payload.get("duration", 0))
        end_tick = int(payload.get("tick", 0))
        payload["end_tick"] = end_tick
        payload["start_tick"] = end_tick - duration
        events.append(payload)
    return events


def _events_for_sync(events, sync_indicator):
    return [
        event
        for event in events
        if int(event.get("sync_indicator", -1)) == sync_indicator
    ]


def _ticks_to_cycles(ticks):
    return int(ticks) // TICKS_PER_CYCLE


def _stage_span_cycles(events):
    if not events:
        raise ValueError("Missing compare profile events")
    start_tick = min(int(event["start_tick"]) for event in events)
    end_tick = max(int(event["end_tick"]) for event in events)
    return _ticks_to_cycles(end_tick - start_tick)


def _busy_cycles(events):
    return _ticks_to_cycles(
        sum(int(event.get("duration", 0)) for event in events)
    )


def _compute_cycles(events):
    return _ticks_to_cycles(
        sum(
            int(event.get("duration", 0))
            for event in events
            if event.get("macro_kind") == "Exec"
            and int(event.get("opcode", -1)) == 3
        )
    )


def emit_compare_perf_summary(active_mpu, scenario):
    if not PROFILE_LOG.is_file():
        raise FileNotFoundError(f"Missing raw profile log: {PROFILE_LOG}")

    events = _parse_profile_end_events(PROFILE_LOG)
    probe_events = _events_for_sync(events, PROBE_SYNC_INDICATOR)
    qk_events = _events_for_sync(events, COMPARE_QK_SYNC_INDICATOR)
    pv_events = _events_for_sync(events, COMPARE_PV_SYNC_INDICATOR)
    compare_events = qk_events + pv_events
    if not probe_events:
        raise RuntimeError("Probe profile events were not captured")
    if not qk_events or not pv_events:
        raise RuntimeError("Compare profile events were not captured")

    compare_cycles = _stage_span_cycles(qk_events) + _stage_span_cycles(
        pv_events
    )
    compare_busy = int(active_mpu.busyCycles())
    compare_latency = _compute_cycles(compare_events)
    compare_idle = compare_cycles - compare_busy
    if compare_idle < 0:
        raise RuntimeError(
            f"Derived negative compare idle cycles: {compare_idle}"
        )

    print(
        "FLASH_ATTENTION_H100_1SM_COMPARE_PERF_SOURCE "
        "source_line=profile_and_MPU_SUMMARY "
        "cycles_field=profile_stage_span_cycles "
        "latency_field=profile_compute_exec_cycles "
        "macs_field=mpu_summary_total_macs "
        "busy_field=mpu_summary_busy "
        "idle_field=profile_stage_idle "
        "spm_wait_field=mpu_summary_spm_wait"
    )
    print(
        "FLASH_ATTENTION_H100_1SM_COMPARE_PERF_SUMMARY "
        f"scenario={scenario} "
        f"compare_cycles={compare_cycles} "
        f"compare_latency={compare_latency} "
        f"compare_macs={active_mpu.totalMacOps()} "
        f"compare_busy={compare_busy} "
        f"compare_idle={compare_idle} "
        f"compare_spm_wait={active_mpu.stallCyclesWaitingForSpm()} "
        "compare_scope=full_flash_attention "
        "softmax_scope=cpu_helper "
        "excluded_from_compare_latency=true "
        "status=PASS"
    )


def build_system(binary, scenario):
    if scenario != SCENARIO:
        raise ValueError(f"Unsupported scenario: {scenario}")

    mem_ranges = [
        AddrRange(0, size=0x60000000),
        AddrRange(0x60000000, size=SPM_SIZE_BYTES),
    ]
    builder = NPUTestSystemBuilder(
        clock=SYSTEM_CLOCK,
        mem_ranges=mem_ranges,
        addr_map=NPUAddressMap(),
    )
    builder.build_base_system()
    builder.add_lowmem(
        AddrRange(0, size=0x60000000),
        latency=DRAM_LATENCY,
        bandwidth=DRAM_BANDWIDTH,
        attr_name="lowmem",
    )
    builder.add_spm(
        size=SPM_SIZE_BYTES,
        latency=SPM_LATENCY,
        bandwidth=SPM_BANDWIDTH,
    )
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(os.path.abspath(binary), argv=[scenario])
    builder.add_megacmdqueue()
    builder.add_mpu(
        attr_name="mpu",
        device_id=0,
        num_mem_side_ports=MPU_MEM_SIDE_PORTS,
        array_dim=MPU_ARRAY_DIM,
        a_buffer_capacity_bytes=MPU_A_BUFFER_BYTES,
        b_buffer_capacity_bytes=MPU_B_BUFFER_BYTES,
        c_buffer_capacity_bytes=MPU_C_BUFFER_BYTES,
    )
    builder.instantiate_root()
    return builder, process


def emit_h100_approximation_summary():
    print(
        "FLASH_ATTENTION_H100_1SM_COMPARE_SYSTEM "
        f"clock={SYSTEM_CLOCK} "
        f"spm_size_bytes={SPM_SIZE_BYTES} "
        f"spm_latency={SPM_LATENCY} "
        f"spm_bandwidth={SPM_BANDWIDTH} "
        f"dram_bandwidth={DRAM_BANDWIDTH} "
        f"dram_map_size_bytes={DRAM_MAP_SIZE_BYTES}"
    )
    print(
        "FLASH_ATTENTION_H100_1SM_COMPARE_TENSOR_APPROX "
        f"array_dim={MPU_ARRAY_DIM} "
        f"a_buffer_bytes={MPU_A_BUFFER_BYTES} "
        f"b_buffer_bytes={MPU_B_BUFFER_BYTES} "
        f"c_buffer_bytes={MPU_C_BUFFER_BYTES} "
        f"mem_side_ports={MPU_MEM_SIDE_PORTS}"
    )
    print(
        "FLASH_ATTENTION_H100_1SM_COMPARE_VECTOR_APPROX "
        "compute_model=shared_riscv_cpu_only "
        f"shared_spm_size_bytes={SPM_SIZE_BYTES} "
        f"dram_bandwidth={DRAM_BANDWIDTH}"
    )


parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--scenario", required=True)
args = parser.parse_args()

builder, process = build_system(args.binary, args.scenario)
m5.instantiate()

builder.map_cmdq(process=process)
builder.map_sync(process=process)
builder.map_spm(process=process)
builder.map_dram(process=process, size=DRAM_MAP_SIZE_BYTES)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()
active_mpu = builder.system.mpu

emit_h100_approximation_summary()
emit_compare_perf_summary(active_mpu, args.scenario)
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
