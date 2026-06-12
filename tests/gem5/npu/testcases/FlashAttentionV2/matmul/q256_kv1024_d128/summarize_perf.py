#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

SHAPE_PREFIX = "FLASH_ATTENTION_MATMUL_SHAPE"
BASELINE_PREFIX = "FLASH_ATTENTION_MATMUL_BASELINE"
FAST_PREFIX = "FLASH_ATTENTION_MATMUL_FAST"
COMPARE_PREFIX = "FLASH_ATTENTION_MATMUL_COMPARE"
PROFILE_BASELINE_PREFIX = "FLASH_ATTENTION_MATMUL_PROFILE_BASELINE"
PROFILE_FAST_PREFIX = "FLASH_ATTENTION_MATMUL_PROFILE_FAST"
PROFILE_COMPARE_PREFIX = "FLASH_ATTENTION_MATMUL_PROFILE_COMPARE"
MPU_SUMMARY_PREFIX = "MPU_SUMMARY"
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
BUFFER_WORD = 5
M_WORD = 6
N_WORD = 7
K_WORD = 8
STRIDE_WORD = 11


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simout", required=True)
    parser.add_argument("--profile-log", required=True)
    parser.add_argument("--profile-json")
    parser.add_argument("--profile-html")
    parser.add_argument("--stdout-artifact")
    parser.add_argument("--output")
    return parser.parse_args()


def extract_kv_line(text, prefix):
    for line in reversed(text.splitlines()):
        if not line.startswith(prefix):
            continue
        return {
            key: value
            for key, value in KEY_VALUE_PATTERN.findall(
                line[len(prefix) :].strip()
            )
        }
    raise ValueError(f"Missing line with prefix: {prefix}")


def require_int(mapping, key):
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"Missing field: {key}")
    return int(value)


def parse_profile_from_json(profile_json_path):
    profile_data = json.loads(profile_json_path.read_text(encoding="utf-8"))
    events = profile_data.get("events", [])
    return {
        "schema": profile_data.get("schema", "unavailable"),
        "source": profile_data.get("source", str(profile_json_path)),
        "axis_count": profile_data.get("axis_count", "unavailable"),
        "event_count": profile_data.get("event_count", len(events)),
        "macro_kind_counts": dict(
            sorted(Counter(event["macro_kind"] for event in events).items())
        ),
    }


def parse_profile_from_raw(profile_log_path):
    macro_kind_counts = Counter()
    event_count = 0

    for line in profile_log_path.read_text(encoding="utf-8").splitlines():
        if "NPU_PROFILE " not in line:
            continue
        payload = json.loads(line.split("NPU_PROFILE ", 1)[1])
        if payload.get("event") != "end":
            continue
        macro_kind_counts[payload.get("macro_kind", "unknown")] += 1
        event_count += 1

    return {
        "schema": "npu-profile-raw-log",
        "source": str(profile_log_path),
        "axis_count": "unavailable",
        "event_count": event_count,
        "macro_kind_counts": dict(sorted(macro_kind_counts.items())),
    }


def parse_profile_end_events(profile_log_path):
    events = []
    for line in profile_log_path.read_text(encoding="utf-8").splitlines():
        if "NPU_PROFILE " not in line:
            continue
        payload = json.loads(line.split("NPU_PROFILE ", 1)[1])
        if payload.get("event") != "end":
            continue
        events.append(payload)
    return events


def summarize_mem_uops(events, sync_indicator):
    phase_events = [
        event
        for event in events
        if int(event.get("sync_indicator", -1)) == sync_indicator
    ]
    mvin_load_uops = [
        int(event.get("issued_load_uops", 0))
        for event in phase_events
        if int(event.get("opcode", -1)) == 0
    ]
    mvout_store_uops = [
        int(event.get("issued_store_uops", 0))
        for event in phase_events
        if int(event.get("opcode", -1)) == 1
    ]
    contiguous_load_uops = []
    contiguous_store_uops = []
    for event in phase_events:
        raw_words = event.get("raw_words", [])
        if len(raw_words) <= STRIDE_WORD:
            continue
        opcode = int(event.get("opcode", -1))
        buffer_kind = int(raw_words[BUFFER_WORD]) & 0x3
        n = int(raw_words[N_WORD])
        k = int(raw_words[K_WORD])
        stride = int(raw_words[STRIDE_WORD])
        if opcode == 0:
            if buffer_kind == 0:
                row_bytes = k
            elif buffer_kind == 1:
                row_bytes = n
            else:
                continue
            if stride == row_bytes:
                contiguous_load_uops.append(
                    int(event.get("issued_load_uops", 0))
                )
        elif opcode == 1:
            if buffer_kind != 2:
                continue
            row_bytes = n * 4
            if stride == row_bytes:
                contiguous_store_uops.append(
                    int(event.get("issued_store_uops", 0))
                )
    return {
        "mvin_macro_count": len(mvin_load_uops),
        "mvout_macro_count": len(mvout_store_uops),
        "fused_compute_macro_count": sum(
            1 for event in phase_events
            if int(event.get("opcode", -1)) == 5
        ),
        "fused_matmul_macro_count": sum(
            1 for event in phase_events
            if int(event.get("opcode", -1)) == 6
        ),
        "mvin_max_load_uops": max(mvin_load_uops, default=0),
        "mvout_max_store_uops": max(mvout_store_uops, default=0),
        "contiguous_load_macro_count": len(contiguous_load_uops),
        "contiguous_store_macro_count": len(contiguous_store_uops),
        "contiguous_load_max_uops": max(contiguous_load_uops, default=0),
        "contiguous_store_max_uops": max(contiguous_store_uops, default=0),
    }


def summarize_macro_start_gaps(profile_json_path, sync_indicator):
    if profile_json_path is None or not profile_json_path.is_file():
        return {
            "macro_count": 0,
            "avg_start_gap_cycles": 0,
            "max_start_gap_cycles": 0,
            "p50_start_gap_cycles": 0,
        }

    profile_data = json.loads(profile_json_path.read_text(encoding="utf-8"))
    events = [
        event
        for event in profile_data.get("events", [])
        if event.get("begin", {}).get("sync_indicator") == sync_indicator
        and "uop" not in event
    ]
    events.sort(key=lambda event: int(event["start_tick"]))
    if len(events) < 2:
        return {
            "macro_count": len(events),
            "avg_start_gap_cycles": 0,
            "max_start_gap_cycles": 0,
            "p50_start_gap_cycles": 0,
        }

    gaps = [
        (
            int(events[idx]["start_tick"]) -
            int(events[idx - 1]["start_tick"])
        )
        // 1000
        for idx in range(1, len(events))
    ]
    gaps.sort()
    return {
        "macro_count": len(events),
        "avg_start_gap_cycles": sum(gaps) // len(gaps),
        "max_start_gap_cycles": gaps[-1],
        "p50_start_gap_cycles": gaps[len(gaps) // 2],
    }


def summarize_sorted_values(values):
    values.sort()
    return {
        "avg_cycles": sum(values) // len(values) if values else 0,
        "max_cycles": values[-1] if values else 0,
        "p50_cycles": values[len(values) // 2] if values else 0,
        "p90_cycles": values[(len(values) * 9) // 10] if values else 0,
    }


def summarize_tick_gaps(ticks):
    ticks.sort()
    gaps = [
        (ticks[idx] - ticks[idx - 1]) // 1000
        for idx in range(1, len(ticks))
    ]
    return {
        "count": len(ticks),
        "gap_count": len(gaps),
        **summarize_sorted_values(gaps),
    }


def parse_launch_profile(profile_log_path, request_sync_indicator=None):
    launches = {}
    request_ticks = []
    request_ticks_all = []
    for line in profile_log_path.read_text(encoding="utf-8").splitlines():
        if "NPU_LAUNCH_PROFILE " not in line:
            continue
        payload = json.loads(line.split("NPU_LAUNCH_PROFILE ", 1)[1])
        if "launch_seq" not in payload:
            continue
        seq = int(payload["launch_seq"])
        launches.setdefault(seq, {})[payload["event"]] = payload
        if payload["event"] == "cpu_launch_request":
            request_tick = int(payload["tick"])
            request_ticks_all.append(request_tick)
            if (
                request_sync_indicator is None or
                int(payload.get("sync_indicator", -1)) ==
                request_sync_indicator
            ):
                request_ticks.append(request_tick)

    latencies = []
    for stages in launches.values():
        if "cpu_launch_request" not in stages:
            continue
        if "cpu_launch_response" not in stages:
            continue
        latencies.append(
            (
                int(stages["cpu_launch_response"]["tick"]) -
                int(stages["cpu_launch_request"]["tick"])
            )
            // 1000
        )
    return {
        "response_latency": {
            "count": len(latencies),
            **summarize_sorted_values(latencies),
        },
        "request_gaps": summarize_tick_gaps(request_ticks),
        "request_gaps_all": summarize_tick_gaps(request_ticks_all),
    }


def main():
    args = parse_args()
    simout_path = Path(args.simout)
    profile_log_path = Path(args.profile_log)
    profile_json_path = Path(args.profile_json) if args.profile_json else None
    profile_html_path = Path(args.profile_html) if args.profile_html else None
    stdout_artifact_path = (
        Path(args.stdout_artifact) if args.stdout_artifact else None
    )
    output_path = Path(args.output) if args.output else None

    if not simout_path.is_file():
        raise FileNotFoundError(f"Missing simout file: {simout_path}")
    if not profile_log_path.is_file():
        raise FileNotFoundError(f"Missing profile log: {profile_log_path}")

    if stdout_artifact_path is not None:
        stdout_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(simout_path, stdout_artifact_path)
        simout_source_path = stdout_artifact_path
    else:
        simout_source_path = simout_path

    simout_text = simout_source_path.read_text(encoding="utf-8")
    shape = extract_kv_line(simout_text, SHAPE_PREFIX)
    baseline = extract_kv_line(simout_text, BASELINE_PREFIX)
    fast = extract_kv_line(simout_text, FAST_PREFIX)
    compare = extract_kv_line(simout_text, COMPARE_PREFIX)
    profile_baseline = extract_kv_line(simout_text, PROFILE_BASELINE_PREFIX)
    profile_fast = extract_kv_line(simout_text, PROFILE_FAST_PREFIX)
    profile_compare = extract_kv_line(simout_text, PROFILE_COMPARE_PREFIX)
    mpu_summary = extract_kv_line(simout_text, MPU_SUMMARY_PREFIX)

    if profile_json_path is not None and profile_json_path.is_file():
        profiling = parse_profile_from_json(profile_json_path)
    else:
        profiling = parse_profile_from_raw(profile_log_path)
    profile_events = parse_profile_end_events(profile_log_path)
    baseline_sync_indicator = require_int(baseline, "sync_indicator")
    fast_sync_indicator = require_int(fast, "sync_indicator")
    total_flops = require_int(shape, "total_flops")
    fast_span_cycles = require_int(profile_fast, "total_span_cycles")
    fast_effective_tops = (
        total_flops / (fast_span_cycles / 1_000_000_000) /
        1_000_000_000_000
    )
    fast_utilization_16tops = fast_effective_tops / 16.0
    launch_profile = parse_launch_profile(
        profile_log_path, fast_sync_indicator
    )

    summary = {
        "m": require_int(shape, "m"),
        "n": require_int(shape, "n"),
        "k": require_int(shape, "k"),
        "tile_m": require_int(shape, "tile_m"),
        "tile_n": require_int(shape, "tile_n"),
        "tile_k": require_int(shape, "tile_k"),
        "total_flops": total_flops,
        "fast_effective_tops": fast_effective_tops,
        "fast_utilization_16tops": fast_utilization_16tops,
        "baseline": {
            "cmds": require_int(baseline, "cmds"),
            "build_cmd_calls": require_int(baseline, "build_cmd_calls"),
            "sync_indicator": require_int(baseline, "sync_indicator"),
            "profile": {
                "total_span_cycles": require_int(
                    profile_baseline, "total_span_cycles"
                ),
                "busy_cycles": require_int(profile_baseline, "busy_cycles"),
                "compute_cycles": require_int(
                    profile_baseline, "compute_cycles"
                ),
                "macro_count": require_int(profile_baseline, "macro_count"),
                **summarize_mem_uops(
                    profile_events, baseline_sync_indicator
                ),
            },
        },
        "fast": {
            "cmds": require_int(fast, "cmds"),
            "template_builds": require_int(fast, "template_builds"),
            "sync_indicator": fast_sync_indicator,
            "profile": {
                "total_span_cycles": require_int(
                    profile_fast, "total_span_cycles"
                ),
                "busy_cycles": require_int(profile_fast, "busy_cycles"),
                "compute_cycles": require_int(profile_fast, "compute_cycles"),
                "macro_count": require_int(profile_fast, "macro_count"),
                **summarize_mem_uops(profile_events, fast_sync_indicator),
            },
        },
        "compare": {
            "build_call_reduction": require_int(
                compare, "build_call_reduction"
            ),
            "baseline_checksum": require_int(compare, "baseline_checksum"),
            "fast_checksum": require_int(compare, "fast_checksum"),
            "ref_checksum": require_int(compare, "ref_checksum"),
            "baseline_total_span_cycles": require_int(
                profile_compare, "baseline_total_span_cycles"
            ),
            "fast_total_span_cycles": require_int(
                profile_compare, "fast_total_span_cycles"
            ),
        },
        "mpu_summary": {
            "cmds": require_int(mpu_summary, "cmds"),
            "compute_cycles": require_int(mpu_summary, "compute_cycles"),
            "drain_cycles": require_int(mpu_summary, "drain_cycles"),
            "output_ready_cycles": require_int(
                mpu_summary, "output_ready_cycles"
            ),
            "cmd_cycles": require_int(mpu_summary, "cmd_cycles"),
            "spm_wait": require_int(mpu_summary, "spm_wait"),
            "macs": require_int(mpu_summary, "macs"),
            "busy": require_int(mpu_summary, "busy"),
            "idle": require_int(mpu_summary, "idle"),
        },
        "fast_issue_gaps": summarize_macro_start_gaps(
            profile_json_path, fast_sync_indicator
        ),
        "launch_profile": launch_profile["response_latency"],
        "launch_request_gaps": launch_profile["request_gaps"],
        "launch_request_gaps_all": launch_profile["request_gaps_all"],
        "profiling": profiling,
        "artifacts": {
            "simout": str(simout_source_path),
            "profile_log": str(profile_log_path),
            "profile_json": (
                str(profile_json_path)
                if profile_json_path is not None and profile_json_path.exists()
                else "unavailable"
            ),
            "profile_html": (
                str(profile_html_path)
                if profile_html_path is not None and profile_html_path.exists()
                else "unavailable"
            ),
        },
    }

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
