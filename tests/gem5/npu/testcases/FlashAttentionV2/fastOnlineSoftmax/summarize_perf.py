#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import re
import shutil
from pathlib import Path

SCENARIO_PREFIX = "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_SCENARIO"
SHAPE_PREFIX = "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_SHAPE"
CMD_PREFIX = "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_CMD"
ACCURACY_PREFIX = "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_ACCURACY"
PROFILE_PREFIX = "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PROFILE"
VPU_SUMMARY_PREFIX = "VPU_SUMMARY"
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


def extract_value(text, prefix):
    for line in reversed(text.splitlines()):
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"Missing line with prefix: {prefix}")


def extract_kv_line(text, prefix):
    for line in reversed(text.splitlines()):
        if line.startswith(prefix):
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


def require_float(mapping, key):
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"Missing field: {key}")
    return float(value)


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
    scenario = extract_value(simout_text, SCENARIO_PREFIX)
    shape = extract_kv_line(simout_text, SHAPE_PREFIX)
    cmd = extract_kv_line(simout_text, CMD_PREFIX)
    accuracy = extract_kv_line(simout_text, ACCURACY_PREFIX)
    profile = extract_kv_line(simout_text, PROFILE_PREFIX)
    vpu_summary = extract_kv_line(simout_text, VPU_SUMMARY_PREFIX)

    summary = {
        "scenario": scenario,
        "shape": {
            "rows": require_int(shape, "rows"),
            "cols": require_int(shape, "cols"),
            "total_elements": require_int(shape, "total_elements"),
            "scratch_slots": require_int(shape, "scratch_slots"),
        },
        "cmd": {
            "template_builds": require_int(cmd, "templates"),
            "launched_cmds": require_int(cmd, "launched"),
            "slice_count": require_int(cmd, "slices"),
            "sync_indicator": require_int(cmd, "sync_indicator"),
        },
        "accuracy": {
            "max_abs_m": require_float(accuracy, "max_abs_m"),
            "max_abs_l": require_float(accuracy, "max_abs_l"),
            "max_abs_p": require_float(accuracy, "max_abs_p"),
            "sample_checksum": require_float(accuracy, "sample_checksum"),
        },
        "profile": {
            "total_span_cycles": require_int(profile, "total_span_cycles"),
            "busy_cycles": require_int(profile, "busy_cycles"),
            "macro_count": require_int(profile, "macro_count"),
            "opcode_counts": {
                "reduce_max": require_int(profile, "reduce_max"),
                "add": require_int(profile, "add"),
                "sub": require_int(profile, "sub"),
                "mul": require_int(profile, "mul"),
                "div": require_int(profile, "div"),
                "scale": require_int(profile, "scale"),
                "abs": require_int(profile, "abs"),
                "reduce_sum": require_int(profile, "reduce_sum"),
                "exp": require_int(profile, "exp"),
            },
        },
        "vpu_summary": {
            "cmds": require_int(vpu_summary, "cmds"),
            "queue_occupancy": require_int(vpu_summary, "queue_occupancy"),
            "issue_busy": vpu_summary["issue_busy"] == "True",
            "executes": require_int(vpu_summary, "executes"),
            "max_active_uops": require_int(vpu_summary, "max_active_uops"),
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
    }

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
