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
TICKS_PER_MHZ_CYCLE = 1_000_000
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


def _ticks_to_cycles(ticks, clock_mhz):
    return int(int(ticks) * float(clock_mhz) / TICKS_PER_MHZ_CYCLE)


def _summarize_phase(events, sync_indicator, clock_mhz):
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
    total_span_ticks = max(event["end_tick"] for event in phase_events) - min(
        event["start_tick"] for event in phase_events
    )
    busy_ticks = sum(int(event.get("duration", 0)) for event in phase_events)
    return {
        "macro_count": len(phase_events),
        "total_span_cycles": _ticks_to_cycles(total_span_ticks, clock_mhz),
        "busy_cycles": _ticks_to_cycles(busy_ticks, clock_mhz),
        "opcode_counts": opcode_counts,
        "start_tick": min(event["start_tick"] for event in phase_events),
        "end_tick": max(event["end_tick"] for event in phase_events),
    }


def emit_profile_summary(clock_mhz):
    if not PROFILE_LOG.is_file():
        print("FLASH_ATTENTION_V2_PROFILE status=FAIL reason=missing_log")
        return False

    events = _parse_profile_end_events(PROFILE_LOG)
    qk = _summarize_phase(events, QK_SYNC_INDICATOR, clock_mhz)
    softmax = _summarize_phase(events, SOFTMAX_SYNC_INDICATOR, clock_mhz)
    pv = _summarize_phase(events, PV_SYNC_INDICATOR, clock_mhz)
    total_span_cycles = _ticks_to_cycles(
        max(qk["end_tick"], softmax["end_tick"], pv["end_tick"])
        - min(qk["start_tick"], softmax["start_tick"], pv["start_tick"]),
        clock_mhz,
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


def _load_simulation_config(path):
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def _nested(config, section, field, default):
    return config.get(section, {}).get(field, default)


def build_system(binary, scenario, q, kv, d, br, bc, simulation):
    if scenario != SCENARIO:
        raise ValueError(f"Unsupported scenario: {scenario}")

    spm_size = _nested(simulation, "spm", "size_bytes", SPM_SIZE_BYTES)
    spm_base = _nested(simulation, "spm", "base_address", 0x60000000)
    xbar_width = _nested(
        simulation, "interconnect", "width_bytes", SYSTEM_XBAR_WIDTH_BYTES
    )
    xbar_latency = _nested(
        simulation,
        "interconnect",
        "latency_cycles",
        SYSTEM_XBAR_LATENCY_CYCLES,
    )
    mem_ranges = [
        AddrRange(0, size=spm_base),
        AddrRange(spm_base, size=spm_size),
    ]
    builder = NPUTestSystemBuilder(
        clock=f"{simulation.get('clock_mhz', 1000)}MHz",
        mem_ranges=mem_ranges,
        addr_map=NPUAddressMap(),
    )
    builder.build_base_system(
        membus_kwargs={
            "width": xbar_width,
            "frontend_latency": xbar_latency,
            "forward_latency": xbar_latency,
            "response_latency": xbar_latency,
            "snoop_response_latency": xbar_latency,
            "header_latency": xbar_latency,
        },
        npu_mmio_bus_kwargs={
            "width": xbar_width,
            "frontend_latency": xbar_latency,
            "forward_latency": xbar_latency,
            "response_latency": xbar_latency,
            "header_latency": xbar_latency,
        },
        cpu_npu_mmio_bus_kwargs={
            "width": xbar_width,
            "frontend_latency": xbar_latency,
            "forward_latency": xbar_latency,
            "response_latency": xbar_latency,
            "header_latency": xbar_latency,
        },
    )
    builder.add_lowmem(
        AddrRange(0, size=spm_base),
        latency=_nested(simulation, "memory", "latency", DRAM_LATENCY),
        bandwidth=_nested(simulation, "memory", "bandwidth", DRAM_BANDWIDTH),
        attr_name="lowmem",
    )
    builder.add_spm(
        size=spm_size,
        latency=_nested(simulation, "spm", "latency", SPM_LATENCY),
        bandwidth=_nested(simulation, "spm", "bandwidth", SPM_BANDWIDTH),
        pipeline_depth=_nested(
            simulation, "spm", "pipeline_depth", SPM_PIPELINE_DEPTH
        ),
        pipeline_ports=_nested(
            simulation, "spm", "pipeline_ports", SPM_PIPELINE_PORTS
        ),
        pipeline_port_stride=_nested(
            simulation,
            "spm",
            "pipeline_port_stride",
            SPM_PIPELINE_PORT_STRIDE,
        ),
    )
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(
        os.path.abspath(binary),
        argv=[
            "--scenario",
            scenario,
            "--q",
            str(q),
            "--kv",
            str(kv),
            "--d",
            str(d),
            "--br",
            str(br),
            "--bc",
            str(bc),
            "--array-dim",
            str(_nested(simulation, "mpu", "array_dim", MPU_ARRAY_DIM)),
        ],
    )
    builder.add_megacmdqueue()
    builder.add_mpu(
        attr_name="mpu",
        device_id=0,
        num_mem_side_ports=_nested(
            simulation, "mpu", "mem_side_ports", MPU_MEM_SIDE_PORTS
        ),
        mem_port_outstanding_limit=_nested(
            simulation,
            "mpu",
            "outstanding_limit",
            MPU_MEM_PORT_OUTSTANDING_LIMIT,
        ),
        array_dim=_nested(simulation, "mpu", "array_dim", MPU_ARRAY_DIM),
        a_buffer_capacity_bytes=_nested(
            simulation, "mpu", "a_buffer_bytes", MPU_A_BUFFER_BYTES
        ),
        b_buffer_capacity_bytes=_nested(
            simulation, "mpu", "b_buffer_bytes", MPU_B_BUFFER_BYTES
        ),
        c_buffer_capacity_bytes=_nested(
            simulation, "mpu", "c_buffer_bytes", MPU_C_BUFFER_BYTES
        ),
    )
    lut = builder.add_lut(
        range_reduction_latency=_nested(
            simulation, "lut", "range_reduction_latency", "1ns"
        ),
        lookup_latency=_nested(simulation, "lut", "lookup_latency", "1ns"),
        interpolation_latency=_nested(
            simulation, "lut", "interpolation_latency", "1ns"
        ),
        normalize_latency=_nested(
            simulation, "lut", "normalize_latency", "1ns"
        ),
        dlen_bytes=_nested(simulation, "vpu", "dlen_bytes", 512),
    )
    builder.add_vpu(
        vpu_id=0,
        num_mem_side_ports=_nested(simulation, "vpu", "mem_side_ports", 32),
        input_buffer_count=_nested(
            simulation, "vpu", "input_buffer_count", 24
        ),
        output_buffer_count=_nested(
            simulation, "vpu", "output_buffer_count", 12
        ),
        local_buffer_stride=_nested(
            simulation, "vpu", "local_buffer_stride", 128 * 128 * 4
        ),
        dlen_bytes=_nested(simulation, "vpu", "dlen_bytes", 512),
        float32_cycles_per_dlen=_nested(
            simulation, "vpu", "float32_cycles_per_dlen", 1
        ),
        float16_cycles_per_dlen=_nested(
            simulation, "vpu", "float16_cycles_per_dlen", 2
        ),
        int32_cycles_per_dlen=_nested(
            simulation, "vpu", "int32_cycles_per_dlen", 2
        ),
        lut=lut,
    )
    builder.instantiate_root()
    return builder, process


parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--scenario", required=True)
parser.add_argument("--q", type=int, default=256)
parser.add_argument("--kv", type=int, default=1024)
parser.add_argument("--d", type=int, default=128)
parser.add_argument("--br", type=int, default=128)
parser.add_argument("--bc", type=int, default=128)
parser.add_argument("--profile-log")
parser.add_argument("--simulation-config")
args = parser.parse_args()

if args.q <= 0 or args.kv <= 0 or args.d <= 0:
    parser.error("--q, --kv, and --d must be positive")
if args.br != 128 or args.bc != 128:
    parser.error("--br and --bc must both be 128")
if args.profile_log:
    PROFILE_LOG = Path(args.profile_log).resolve()
simulation = _load_simulation_config(args.simulation_config)
if simulation.get("clock_mhz", 1000) <= 0:
    parser.error("simulation clock must be positive")

builder, process = build_system(
    args.binary,
    args.scenario,
    args.q,
    args.kv,
    args.d,
    args.br,
    args.bc,
    simulation,
)
m5.instantiate()

builder.map_cmdq(process=process)
builder.map_sync(process=process)
builder.map_spm(process=process)
builder.map_dram(
    process=process,
    size=_nested(simulation, "memory", "dram_size_bytes", DRAM_MAP_SIZE_BYTES),
)
builder.map_vpu(process=process, vpu_id=0)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()
active_mpu = builder.system.mpu
active_vpu = builder.system.vpu0
attention_tiles = ((args.q + args.br - 1) // args.br) * (
    (args.kv + args.bc - 1) // args.bc
)

profile_ok = emit_profile_summary(simulation.get("clock_mhz", 1000))
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
    and active_mpu.completedCmdCount() == attention_tiles * 2
    and active_vpu.completedCmdCount() == attention_tiles * 14
    and active_vpu.queueOccupancy() == 0
):
    print("FLASH_ATTENTION_V2_CONFIG_PASS")
