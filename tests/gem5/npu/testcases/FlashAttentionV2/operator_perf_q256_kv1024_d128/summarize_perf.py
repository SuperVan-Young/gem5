#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import re
import shutil
from pathlib import Path

KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
SHAPE_PREFIX = "FLASH_ATTENTION_V2_SHAPE"
PROFILE_QK_PREFIX = "FLASH_ATTENTION_V2_PROFILE_QK"
PROFILE_SOFTMAX_PREFIX = "FLASH_ATTENTION_V2_PROFILE_SOFTMAX"
PROFILE_PV_PREFIX = "FLASH_ATTENTION_V2_PROFILE_PV"
PROFILE_TOTAL_PREFIX = "FLASH_ATTENTION_V2_PROFILE_TOTAL"
MPU_SUMMARY_PREFIX = "FLASH_ATTENTION_V2_MPU_SUMMARY"
VPU_SUMMARY_PREFIX = "FLASH_ATTENTION_V2_VPU_SUMMARY"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simout", required=True)
    parser.add_argument("--profile-log", required=True)
    parser.add_argument("--profile-json")
    parser.add_argument("--profile-html")
    parser.add_argument("--stdout-artifact")
    parser.add_argument("--output")
    parser.add_argument("--clock-mhz", type=float, default=1000.0)
    parser.add_argument("--array-dim", type=int, default=128)
    return parser.parse_args()


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


def stage_summary(mapping):
    return {
        "total_span_cycles": require_int(mapping, "total_span_cycles"),
        "busy_cycles": require_int(mapping, "busy_cycles"),
        "macro_count": require_int(mapping, "macro_count"),
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
    qk = extract_kv_line(simout_text, PROFILE_QK_PREFIX)
    softmax = extract_kv_line(simout_text, PROFILE_SOFTMAX_PREFIX)
    pv = extract_kv_line(simout_text, PROFILE_PV_PREFIX)
    total = extract_kv_line(simout_text, PROFILE_TOTAL_PREFIX)
    mpu_summary = extract_kv_line(simout_text, MPU_SUMMARY_PREFIX)
    vpu_summary = extract_kv_line(simout_text, VPU_SUMMARY_PREFIX)

    total_matmul_flops = require_int(shape, "total_matmul_flops")
    total_span_cycles = require_int(total, "total_span_cycles")
    total_busy_cycles = require_int(total, "busy_cycles")
    clock_hz = args.clock_mhz * 1_000_000
    theoretical_tops = (
        args.array_dim * args.array_dim * clock_hz / 1_000_000_000_000
    )
    host_inclusive_tops = (
        total_matmul_flops / (total_span_cycles / clock_hz) / 1_000_000_000_000
    )
    effective_tops = (
        total_matmul_flops / (total_busy_cycles / clock_hz) / 1_000_000_000_000
    )
    utilization = effective_tops / theoretical_tops

    summary = {
        "shape": {
            "q": require_int(shape, "q"),
            "kv": require_int(shape, "kv"),
            "d": require_int(shape, "d"),
            "matmul_count": require_int(shape, "matmul_count"),
            "single_matmul_flops": require_int(shape, "single_matmul_flops"),
            "total_matmul_flops": total_matmul_flops,
        },
        "profile": {
            "qk_fast_matmul": {
                **stage_summary(qk),
                "fused_matmul": require_int(qk, "fused_matmul"),
            },
            "fast_online_softmax": {
                **stage_summary(softmax),
                "reduce_max": require_int(softmax, "reduce_max"),
                "add": require_int(softmax, "add"),
                "sub": require_int(softmax, "sub"),
                "mul": require_int(softmax, "mul"),
                "div": require_int(softmax, "div"),
                "scale": require_int(softmax, "scale"),
                "abs": require_int(softmax, "abs"),
                "reduce_sum": require_int(softmax, "reduce_sum"),
                "exp": require_int(softmax, "exp"),
            },
            "pv_fast_matmul": {
                **stage_summary(pv),
                "fused_matmul": require_int(pv, "fused_matmul"),
            },
            "operator_total": stage_summary(total),
        },
        "performance": {
            "primary_cycle_metric": "busy_cycles",
            "clock_hz": clock_hz,
            "array_dim": args.array_dim,
            "theoretical_tops": theoretical_tops,
            "total_tensor_macs": total_matmul_flops // 2,
            "total_tensor_ops": total_matmul_flops,
            "op_counting": "multiply_add_as_two_ops",
            "npu_active_cycles": total_busy_cycles,
            "operator_busy_cycles": total_busy_cycles,
            "host_inclusive_cycles": total_span_cycles,
            "operator_elapsed_cycles": total_span_cycles,
            "effective_tops": effective_tops,
            "utilization": utilization,
            "utilization_percent": utilization * 100.0,
            "host_inclusive_effective_tops": host_inclusive_tops,
            "host_inclusive_utilization": (
                host_inclusive_tops / theoretical_tops
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
