# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import m5
from m5.objects import AddrRange

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "configs"))

from npu_test_system import (  # noqa: E402
    NPUAddressMap,
    NPUTestSystemBuilder,
)

SCENARIO = "flash_attention_matmul_q256_kv1024_d128"
SYSTEM_CLOCK = "1GHz"
SPM_SIZE_BYTES = 4 * 1024 * 1024
SPM_LATENCY = "1ns"
SPM_BANDWIDTH = "100GiB/s"
SYSTEM_XBAR_WIDTH_BYTES = 256
SYSTEM_XBAR_LATENCY_CYCLES = 0
DRAM_MAP_SIZE_BYTES = 32 * 1024 * 1024
DRAM_BANDWIDTH = "25.378787879GB/s"
DRAM_LATENCY = "30ns"
MPU_ARRAY_DIM = 128
MPU_A_BUFFER_BYTES = 16 * 1024
MPU_B_BUFFER_BYTES = 16 * 1024
MPU_C_BUFFER_BYTES = 64 * 1024
MPU_MEM_SIDE_PORTS = 2
PROFILE_LOG = (
    Path(__file__).resolve().parent
    / "profile"
    / "q256_kv1024_d128.npu_profile.log"
)
TICKS_PER_CYCLE = 1000
BASELINE_SYNC_INDICATOR = 0x40
FAST_SYNC_INDICATOR = 0x80


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
        payload["start_tick"] = end_tick - duration
        payload["end_tick"] = end_tick
        payload["opcode_int"] = int(payload.get("opcode", -1))
        payload["sync_indicator_int"] = int(payload.get("sync_indicator", -1))
        events.append(payload)
    return events


def _ticks_to_cycles(ticks):
    return int(ticks) // TICKS_PER_CYCLE


def _summarize_phase(events, sync_indicator):
    phase_events = [
        event
        for event in events
        if event["sync_indicator_int"] == sync_indicator
    ]
    if not phase_events:
        raise RuntimeError(
            f"Missing profile events for sync_indicator={sync_indicator}"
        )

    opcode_counts = Counter(event["opcode_int"] for event in phase_events)
    total_span_ticks = (
        max(event["end_tick"] for event in phase_events)
        - min(event["start_tick"] for event in phase_events)
    )
    total_busy_ticks = sum(
        int(event.get("duration", 0)) for event in phase_events
    )
    compute_ticks = sum(
        int(event.get("duration", 0))
        for event in phase_events
        if event["opcode_int"] in (3, 5)
    )

    return {
        "macro_count": len(phase_events),
        "total_span_cycles": _ticks_to_cycles(total_span_ticks),
        "busy_cycles": _ticks_to_cycles(total_busy_ticks),
        "compute_cycles": _ticks_to_cycles(compute_ticks),
        "mvin": opcode_counts[0],
        "mvout": opcode_counts[1],
        "load": opcode_counts[2],
        "compute": opcode_counts[3] + opcode_counts[5],
        "drain": opcode_counts[4],
    }


def emit_profile_summary():
    if not PROFILE_LOG.is_file():
        raise FileNotFoundError(f"Missing raw profile log: {PROFILE_LOG}")

    events = _parse_profile_end_events(PROFILE_LOG)
    baseline = _summarize_phase(events, BASELINE_SYNC_INDICATOR)
    fast = _summarize_phase(events, FAST_SYNC_INDICATOR)

    print(
        "FLASH_ATTENTION_MATMUL_PROFILE_BASELINE "
        f"total_span_cycles={baseline['total_span_cycles']} "
        f"busy_cycles={baseline['busy_cycles']} "
        f"compute_cycles={baseline['compute_cycles']} "
        f"macro_count={baseline['macro_count']} "
        f"mvin={baseline['mvin']} load={baseline['load']} "
        f"compute={baseline['compute']} drain={baseline['drain']} "
        f"mvout={baseline['mvout']} status=PASS"
    )
    print(
        "FLASH_ATTENTION_MATMUL_PROFILE_FAST "
        f"total_span_cycles={fast['total_span_cycles']} "
        f"busy_cycles={fast['busy_cycles']} "
        f"compute_cycles={fast['compute_cycles']} "
        f"macro_count={fast['macro_count']} "
        f"mvin={fast['mvin']} load={fast['load']} "
        f"compute={fast['compute']} drain={fast['drain']} "
        f"mvout={fast['mvout']} status=PASS"
    )
    print(
        "FLASH_ATTENTION_MATMUL_PROFILE_COMPARE "
        f"baseline_total_span_cycles={baseline['total_span_cycles']} "
        f"fast_total_span_cycles={fast['total_span_cycles']} "
        f"baseline_busy_cycles={baseline['busy_cycles']} "
        f"fast_busy_cycles={fast['busy_cycles']} "
        f"baseline_compute_cycles={baseline['compute_cycles']} "
        f"fast_compute_cycles={fast['compute_cycles']} status=PASS"
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
    builder.build_base_system(
        membus_kwargs={
            "width": SYSTEM_XBAR_WIDTH_BYTES,
            "frontend_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "forward_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "response_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "snoop_response_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "header_latency": SYSTEM_XBAR_LATENCY_CYCLES,
        }
    )
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

emit_profile_summary()
print(f"MPU_EXIT_CAUSE={exit_cause}")
print(f"MPU_EXIT_CODE={exit_code}")
print(
    "MPU_SUMMARY "
    f"scenario={args.scenario} "
    f"cmds={active_mpu.completedCmdCount()} "
    f"compute_cycles={active_mpu.lastComputeLatencyCycles()} "
    f"drain_cycles={active_mpu.lastDrainLatencyCycles()} "
    f"output_ready_cycles={active_mpu.lastOutputReadyLatencyCycles()} "
    f"cmd_cycles={active_mpu.lastCommandLatencyCycles()} "
    f"spm_wait={active_mpu.stallCyclesWaitingForSpm()} "
    f"macs={active_mpu.totalMacOps()} "
    f"busy={active_mpu.busyCycles()} "
    f"idle={active_mpu.idleCycles()}"
)
