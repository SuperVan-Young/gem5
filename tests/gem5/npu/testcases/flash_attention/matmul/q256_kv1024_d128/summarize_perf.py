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

    summary = {
        "m": require_int(shape, "m"),
        "n": require_int(shape, "n"),
        "k": require_int(shape, "k"),
        "tile_m": require_int(shape, "tile_m"),
        "tile_n": require_int(shape, "tile_n"),
        "tile_k": require_int(shape, "tile_k"),
        "total_flops": require_int(shape, "total_flops"),
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
            },
        },
        "fast": {
            "cmds": require_int(fast, "cmds"),
            "template_builds": require_int(fast, "template_builds"),
            "sync_indicator": require_int(fast, "sync_indicator"),
            "profile": {
                "total_span_cycles": require_int(
                    profile_fast, "total_span_cycles"
                ),
                "busy_cycles": require_int(profile_fast, "busy_cycles"),
                "compute_cycles": require_int(profile_fast, "compute_cycles"),
                "macro_count": require_int(profile_fast, "macro_count"),
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
            "cmd_cycles": require_int(mpu_summary, "cmd_cycles"),
            "spm_wait": require_int(mpu_summary, "spm_wait"),
            "macs": require_int(mpu_summary, "macs"),
            "busy": require_int(mpu_summary, "busy"),
            "idle": require_int(mpu_summary, "idle"),
        },
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
