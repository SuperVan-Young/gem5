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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_system import (  # noqa: E402
    NPUAddressMap,
    NPUTestSystemBuilder,
)

SCENARIO = "flash_attention_v2_operator_perf_q256_kv1024_d128"
SYSTEM_CLOCK = "1GHz"
SPM_SIZE_BYTES = 8 * 1024 * 1024
SPM_LATENCY = "1ns"
SPM_BANDWIDTH = "100GiB/s"
SPM_PIPELINE_DEPTH = 1024
SPM_PIPELINE_PORTS = 32
SPM_PIPELINE_PORT_STRIDE = 64
SYSTEM_XBAR_WIDTH_BYTES = 256
SYSTEM_XBAR_LATENCY_CYCLES = 0
DRAM_MAP_SIZE_BYTES = 32 * 1024 * 1024
DRAM_BANDWIDTH = "25.378787879GB/s"
DRAM_LATENCY = "30ns"
MPU_ARRAY_DIM = 128
MPU_A_BUFFER_BYTES = 16 * 1024
MPU_B_BUFFER_BYTES = 16 * 1024
MPU_C_BUFFER_BYTES = 64 * 1024
MPU_MEM_SIDE_PORTS = 3
MPU_MEM_PORT_OUTSTANDING_LIMIT = 1024
PROFILE_LOG = (
    Path(__file__).resolve().parent
    / "profile"
    / "flashAttentionV2.npu_profile.log"
)
TICKS_PER_CYCLE = 1000
QK_SYNC_INDICATOR = 0x81
SOFTMAX_SYNC_INDICATOR = 0x91
PV_SYNC_INDICATOR = 0x82
EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0


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
    busy_ticks = sum(int(event.get("duration", 0)) for event in phase_events)
    return {
        "macro_count": len(phase_events),
        "total_span_cycles": _ticks_to_cycles(total_span_ticks),
        "busy_cycles": _ticks_to_cycles(busy_ticks),
        "opcode_counts": opcode_counts,
        "start_tick": min(event["start_tick"] for event in phase_events),
        "end_tick": max(event["end_tick"] for event in phase_events),
    }


def emit_profile_summary():
    if not PROFILE_LOG.is_file():
        print("FLASH_ATTENTION_V2_PROFILE status=FAIL reason=missing_log")
        return False

    events = _parse_profile_end_events(PROFILE_LOG)
    qk = _summarize_phase(events, QK_SYNC_INDICATOR)
    softmax = _summarize_phase(events, SOFTMAX_SYNC_INDICATOR)
    pv = _summarize_phase(events, PV_SYNC_INDICATOR)
    total_span_cycles = _ticks_to_cycles(
        max(qk["end_tick"], softmax["end_tick"], pv["end_tick"]) -
        min(qk["start_tick"], softmax["start_tick"], pv["start_tick"])
    )

    print(
        "FLASH_ATTENTION_V2_PROFILE_QK "
        f"total_span_cycles={qk['total_span_cycles']} "
        f"busy_cycles={qk['busy_cycles']} "
        f"macro_count={qk['macro_count']} "
        f"fused_matmul={qk['opcode_counts'][6]} status=PASS"
    )
    print(
        "FLASH_ATTENTION_V2_PROFILE_SOFTMAX "
        f"total_span_cycles={softmax['total_span_cycles']} "
        f"busy_cycles={softmax['busy_cycles']} "
        f"macro_count={softmax['macro_count']} "
        f"reduce_max={softmax['opcode_counts'][11]} "
        f"add={softmax['opcode_counts'][1]} "
        f"sub={softmax['opcode_counts'][2]} "
        f"mul={softmax['opcode_counts'][3]} "
        f"div={softmax['opcode_counts'][4]} "
        f"scale={softmax['opcode_counts'][5]} "
        f"abs={softmax['opcode_counts'][9]} "
        f"reduce_sum={softmax['opcode_counts'][10]} "
        f"exp={softmax['opcode_counts'][14]} status=PASS"
    )
    print(
        "FLASH_ATTENTION_V2_PROFILE_PV "
        f"total_span_cycles={pv['total_span_cycles']} "
        f"busy_cycles={pv['busy_cycles']} "
        f"macro_count={pv['macro_count']} "
        f"fused_matmul={pv['opcode_counts'][6]} status=PASS"
    )
    total_busy_cycles = (
        qk["busy_cycles"] + softmax["busy_cycles"] + pv["busy_cycles"]
    )
    total_macro_count = (
        qk["macro_count"] + softmax["macro_count"] + pv["macro_count"]
    )
    print(
        "FLASH_ATTENTION_V2_PROFILE_TOTAL "
        f"total_span_cycles={total_span_cycles} "
        f"busy_cycles={total_busy_cycles} "
        f"macro_count={total_macro_count} "
        "status=PASS"
    )
    return True


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
        },
        npu_mmio_bus_kwargs={
            "width": SYSTEM_XBAR_WIDTH_BYTES,
            "frontend_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "forward_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "response_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "header_latency": SYSTEM_XBAR_LATENCY_CYCLES,
        },
        cpu_npu_mmio_bus_kwargs={
            "width": SYSTEM_XBAR_WIDTH_BYTES,
            "frontend_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "forward_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "response_latency": SYSTEM_XBAR_LATENCY_CYCLES,
            "header_latency": SYSTEM_XBAR_LATENCY_CYCLES,
        },
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
        pipeline_depth=SPM_PIPELINE_DEPTH,
        pipeline_ports=SPM_PIPELINE_PORTS,
        pipeline_port_stride=SPM_PIPELINE_PORT_STRIDE,
    )
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(os.path.abspath(binary), argv=[scenario])
    builder.add_megacmdqueue()
    builder.add_mpu(
        attr_name="mpu",
        device_id=0,
        num_mem_side_ports=MPU_MEM_SIDE_PORTS,
        mem_port_outstanding_limit=MPU_MEM_PORT_OUTSTANDING_LIMIT,
        array_dim=MPU_ARRAY_DIM,
        a_buffer_capacity_bytes=MPU_A_BUFFER_BYTES,
        b_buffer_capacity_bytes=MPU_B_BUFFER_BYTES,
        c_buffer_capacity_bytes=MPU_C_BUFFER_BYTES,
    )
    lut = builder.add_lut(
        range_reduction_latency="1ns",
        lookup_latency="1ns",
        interpolation_latency="1ns",
        normalize_latency="1ns",
        dlen_bytes=512,
    )
    builder.add_vpu(
        vpu_id=0,
        num_mem_side_ports=32,
        input_buffer_count=24,
        output_buffer_count=12,
        local_buffer_stride=128 * 128 * 4,
        dlen_bytes=512,
        float32_cycles_per_dlen=1,
        float16_cycles_per_dlen=2,
        int32_cycles_per_dlen=2,
        lut=lut,
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
builder.map_vpu(process=process, vpu_id=0)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()
active_mpu = builder.system.mpu
active_vpu = builder.system.vpu0

profile_ok = emit_profile_summary()
print(f"FLASH_ATTENTION_V2_EXIT_CAUSE={exit_cause}")
print(f"FLASH_ATTENTION_V2_EXIT_CODE={exit_code}")
print(
    "FLASH_ATTENTION_V2_MPU_SUMMARY "
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
print(
    "FLASH_ATTENTION_V2_VPU_SUMMARY "
    f"cmds={active_vpu.completedCmdCount()} "
    f"queue_occupancy={active_vpu.queueOccupancy()} "
    f"issue_busy={active_vpu.isIssueBusy()} "
    f"executes={active_vpu.executeCount()} "
    f"max_active_uops={active_vpu.maxActiveMicroOps()}"
)

if (
    profile_ok
    and exit_cause == EXPECTED_EXIT_CAUSE
    and exit_code == EXPECTED_EXIT_CODE
    and active_mpu.completedCmdCount() == 2
    and active_vpu.completedCmdCount() == 14
    and active_vpu.queueOccupancy() == 0
):
    print("FLASH_ATTENTION_V2_CONFIG_PASS")
