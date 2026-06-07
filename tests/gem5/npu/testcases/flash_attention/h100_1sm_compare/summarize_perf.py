#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

WORKLOAD_PREFIX = "FLASH_ATTENTION_H100_1SM_COMPARE_WORKLOAD"
PERF_SOURCE_PREFIX = "FLASH_ATTENTION_H100_1SM_COMPARE_PERF_SOURCE"
PERF_SUMMARY_PREFIX = "FLASH_ATTENTION_H100_1SM_COMPARE_PERF_SUMMARY"
MPU_SUMMARY_PREFIX = "MPU_SUMMARY"
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


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


def require_int_value(mapping, key):
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"Missing field: {key}")
    if value == "unavailable":
        raise ValueError(f"Field must be numeric, got unavailable: {key}")
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
        "total_duration": sum(
            int(event.get("duration", 0)) for event in events
        ),
    }


def parse_profile_from_raw(profile_log_path):
    macro_kind_counts = Counter()
    event_count = 0
    total_duration = 0

    for line in profile_log_path.read_text(encoding="utf-8").splitlines():
        if "NPU_PROFILE " not in line:
            continue
        payload = json.loads(line.split("NPU_PROFILE ", 1)[1])
        if payload.get("event") != "end":
            continue
        macro_kind = payload.get("macro_kind", "unknown")
        macro_kind_counts[macro_kind] += 1
        event_count += 1
        total_duration += int(payload.get("duration", 0))

    return {
        "schema": "npu-profile-raw-log",
        "source": str(profile_log_path),
        "axis_count": "unavailable",
        "event_count": event_count,
        "macro_kind_counts": dict(sorted(macro_kind_counts.items())),
        "total_duration": total_duration,
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

    workload = extract_kv_line(simout_text, WORKLOAD_PREFIX)
    perf_source = extract_kv_line(simout_text, PERF_SOURCE_PREFIX)
    perf_summary = extract_kv_line(simout_text, PERF_SUMMARY_PREFIX)
    mpu_summary = extract_kv_line(simout_text, MPU_SUMMARY_PREFIX)

    if profile_json_path is not None and profile_json_path.is_file():
        profiling = parse_profile_from_json(profile_json_path)
    else:
        profiling = parse_profile_from_raw(profile_log_path)

    summary = {
        "scenario": workload["scenario"],
        "seq_len": require_int(workload, "seq_len"),
        "head_dim": require_int(workload, "head_dim"),
        "qk_flops": require_int(workload, "qk_flops"),
        "pv_flops": require_int(workload, "pv_flops"),
        "total_flops": require_int(workload, "total_flops"),
        "compare_cycles": require_int_value(perf_summary, "compare_cycles"),
        "compare_latency": require_int_value(
            perf_summary, "compare_latency"
        ),
        "compare_issue_latency": require_int_value(
            perf_summary, "compare_issue_latency"
        ),
        "compare_completion_latency": require_int_value(
            perf_summary, "compare_completion_latency"
        ),
        "compare_macs": require_int_value(perf_summary, "compare_macs"),
        "compare_busy": require_int_value(perf_summary, "compare_busy"),
        "compare_idle": require_int_value(perf_summary, "compare_idle"),
        "compare_spm_wait": require_int_value(
            perf_summary, "compare_spm_wait"
        ),
        "compare_scope": perf_summary["compare_scope"],
        "softmax_scope": perf_summary["softmax_scope"],
        "excluded_from_compare_latency": (
            perf_summary["excluded_from_compare_latency"] == "true"
        ),
        "perf_source": {
            "source_line": perf_source["source_line"],
            "cycles_field": perf_source["cycles_field"],
            "latency_field": perf_source["latency_field"],
            "macs_field": perf_source["macs_field"],
            "busy_field": perf_source["busy_field"],
            "idle_field": perf_source["idle_field"],
            "spm_wait_field": perf_source["spm_wait_field"],
        },
        "mpu_summary": {
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
        "profiling": profiling,
    }

    if summary["total_flops"] != summary["qk_flops"] + summary["pv_flops"]:
        raise ValueError("total_flops must equal qk_flops + pv_flops")

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
