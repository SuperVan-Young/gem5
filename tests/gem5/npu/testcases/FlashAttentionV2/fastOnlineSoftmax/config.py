# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_system import NPUTestSystemBuilder  # noqa: E402

SCENARIO = "flash_attention_fast_online_softmax_256x1024"
PROFILE_LOG = (
    Path(__file__).resolve().parent
    / "profile"
    / "fastOnlineSoftmax.npu_profile.log"
)
TICKS_PER_CYCLE = 1000
SYNC_INDICATOR = 0x91
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


def emit_profile_summary():
    if not PROFILE_LOG.is_file():
        print(
            "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PROFILE "
            "status=FAIL reason=missing_profile_log"
        )
        return False

    events = [
        event
        for event in _parse_profile_end_events(PROFILE_LOG)
        if event["sync_indicator_int"] == SYNC_INDICATOR
    ]
    if not events:
        print(
            "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PROFILE "
            "status=FAIL reason=missing_profile_events"
        )
        return False

    opcode_counts = Counter(event["opcode_int"] for event in events)
    total_span_ticks = (
        max(event["end_tick"] for event in events)
        - min(event["start_tick"] for event in events)
    )
    busy_ticks = sum(int(event.get("duration", 0)) for event in events)

    print(
        "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PROFILE "
        f"total_span_cycles={_ticks_to_cycles(total_span_ticks)} "
        f"busy_cycles={_ticks_to_cycles(busy_ticks)} "
        f"macro_count={len(events)} "
        f"reduce_max={opcode_counts[11]} add={opcode_counts[1]} "
        f"sub={opcode_counts[2]} mul={opcode_counts[3]} "
        f"div={opcode_counts[4]} scale={opcode_counts[5]} "
        f"abs={opcode_counts[9]} reduce_sum={opcode_counts[10]} "
        f"exp={opcode_counts[14]} status=PASS"
    )
    return True


def build_system(binary, scenario):
    if scenario != SCENARIO:
        raise ValueError(f"Unsupported scenario: {scenario}")

    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_spm(size=8 * 1024 * 1024)
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(os.path.abspath(binary), argv=[scenario])
    builder.add_megacmdqueue()
    lut = builder.add_lut(
        range_reduction_latency="1ns",
        lookup_latency="1ns",
        interpolation_latency="1ns",
        normalize_latency="1ns",
        dlen_bytes=128,
    )
    builder.add_vpu(
        vpu_id=0,
        num_mem_side_ports=32,
        input_buffer_count=24,
        output_buffer_count=12,
        local_buffer_stride=128 * 128 * 4,
        dlen_bytes=128,
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
builder.map_vpu(process=process, vpu_id=0)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()
active_vpu = builder.system.vpu0

emit_profile_summary()
print(f"VPU_EXIT_CAUSE={exit_cause}")
print(f"VPU_EXIT_CODE={exit_code}")
print(
    "VPU_SUMMARY "
    f"scenario={args.scenario} "
    f"cmds={active_vpu.completedCmdCount()} "
    f"queue_occupancy={active_vpu.queueOccupancy()} "
    f"issue_busy={active_vpu.isIssueBusy()} "
    f"executes={active_vpu.executeCount()} "
    f"max_active_uops={active_vpu.maxActiveMicroOps()}"
)

if (
    exit_cause == EXPECTED_EXIT_CAUSE
    and exit_code == EXPECTED_EXIT_CODE
    and active_vpu.queueOccupancy() == 0
    and active_vpu.completedCmdCount() == 14
):
    print("FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_CONFIG_PASS")
