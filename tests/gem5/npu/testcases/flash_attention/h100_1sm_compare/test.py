# Copyright (c) 2026
# All rights reserved.

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from runner_common import (  # noqa: E402
    NpuRunnerSpec,
    make_binary_config_args,
    make_profile_artifact_verifier,
    make_profile_gem5_args,
    make_testcase_build_fixture,
    register_npu_test,
    resolve_binary_path,
    resolve_config_path,
)
from testlib import test_util  # noqa: E402
from testlib.configuration import constants  # noqa: E402

from gem5 import verifier  # noqa: E402

SCENARIO = "flash_attention_h100_1sm_compare"
TESTCASE_ROOT = Path(__file__).resolve().parent
PROFILE_DIR = TESTCASE_ROOT / "profile"
PROFILE_LOG = PROFILE_DIR / "h100_1sm_compare.npu_profile.log"
PROFILE_JSON = PROFILE_DIR / "h100_1sm_compare.npu_profile.json"
PROFILE_HTML = PROFILE_DIR / "h100_1sm_compare.npu_profile.html"
STDOUT_ARTIFACT = PROFILE_DIR / "h100_1sm_compare.simout.txt"
PERF_SUMMARY = PROFILE_DIR / "h100_1sm_compare.performance_summary.json"
PERF_SUMMARIZER = TESTCASE_ROOT / "summarize_perf.py"
binary = resolve_binary_path(
    __file__, "flash_attention_h100_1sm_compare_riscv"
)


class GeneratePerformanceSummary(verifier.Verifier):
    def __init__(self):
        super().__init__()

    def test(self, params):
        tempdir = Path(params.fixtures[constants.tempdir_fixture_name].path)
        simout_path = tempdir / constants.gem5_simulation_stdout
        result = subprocess.run(
            [
                sys.executable,
                str(PERF_SUMMARIZER),
                "--simout",
                str(simout_path),
                "--profile-log",
                str(PROFILE_LOG),
                "--profile-json",
                str(PROFILE_JSON),
                "--profile-html",
                str(PROFILE_HTML),
                "--stdout-artifact",
                str(STDOUT_ARTIFACT),
                "--output",
                str(PERF_SUMMARY),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            test_util.fail(
                "summarize_perf.py failed for %s:\nstdout:\n%s\nstderr:\n%s",
                simout_path,
                result.stdout,
                result.stderr,
            )

        if not PERF_SUMMARY.is_file():
            test_util.fail(
                "Missing performance summary artifact %s", PERF_SUMMARY
            )

        try:
            summary = json.loads(PERF_SUMMARY.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            test_util.fail("Invalid JSON in %s: %s", PERF_SUMMARY, exc)

        required_top_level = (
            "scenario",
            "seq_len",
            "head_dim",
            "qk_flops",
            "pv_flops",
            "total_flops",
            "compare_cycles",
            "compare_latency",
            "compare_macs",
            "compare_busy",
            "compare_idle",
            "compare_spm_wait",
            "compare_scope",
            "softmax_scope",
            "mpu_summary",
            "profiling",
        )
        missing = [key for key in required_top_level if key not in summary]
        if missing:
            test_util.fail(
                "Missing performance summary keys %s in %s",
                ", ".join(missing),
                PERF_SUMMARY,
            )

        required_mpu_fields = (
            "compute_cycles",
            "cmd_cycles",
            "spm_wait",
            "macs",
            "busy",
            "idle",
        )
        missing_mpu = [
            key
            for key in required_mpu_fields
            if key not in summary["mpu_summary"]
        ]
        if missing_mpu:
            test_util.fail(
                "Missing MPU summary keys %s in %s",
                ", ".join(missing_mpu),
                PERF_SUMMARY,
            )

        if summary["scenario"] != SCENARIO:
            test_util.fail(
                "Unexpected scenario %s in %s",
                summary["scenario"],
                PERF_SUMMARY,
            )

        numeric_compare_fields = (
            "compare_cycles",
            "compare_latency",
            "compare_macs",
            "compare_busy",
            "compare_idle",
            "compare_spm_wait",
        )
        for key in numeric_compare_fields:
            if not isinstance(summary[key], int):
                test_util.fail(
                    "Expected integer compare field %s in %s, got %r",
                    key,
                    PERF_SUMMARY,
                    summary[key],
                )

        if summary["total_flops"] != summary["qk_flops"] + summary["pv_flops"]:
            test_util.fail(
                "total_flops mismatch in %s: %s != %s + %s",
                PERF_SUMMARY,
                summary["total_flops"],
                summary["qk_flops"],
                summary["pv_flops"],
            )

        if summary["profiling"].get("event_count", 0) <= 0:
            test_util.fail(
                "Expected profiling.event_count > 0 in %s", PERF_SUMMARY
            )


register_npu_test(
    NpuRunnerSpec(
        name="flash_attention_h100_1sm_compare",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(binary, "--scenario", SCENARIO)
        ),
        gem5_args=make_profile_gem5_args(__file__),
        verifier_specs=(
            rf"FLASH_ATTENTION_H100_1SM_COMPARE_SCENARIO={SCENARIO}",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_SHAPE "
            r"seq_len=1024 head_dim=128 batch=1 heads=1",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_CONTRACT "
            r"sms=1 mapping=one_sm_per_head_per_batch",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_WORKLOAD "
            r"scenario=flash_attention_h100_1sm_compare seq_len=1024 "
            r"head_dim=128 qk_flops=268435456 pv_flops=268435456 "
            r"total_flops=536870912 pass=PASS",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_PROBE "
            r"m=2 n=3 k=4 macs=24 sync_indicator=17 status=PASS",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_PROFILE_TAGS "
            r"probe_sync=17 qk_sync=33 pv_sync=49",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_BEGIN "
            r"scenario=flash_attention_h100_1sm_compare "
            r"compare_scope=full_flash_attention "
            r"softmax_scope=cpu_helper tensor_path=qk_pv_only "
            r"excluded_from_compare_latency=true",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_FLOPS "
            r"qk_flops=268435456 pv_flops=268435456 total_flops=536870912 "
            r"tensor_path_only=true cpu_softmax_included=false",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_OUTPUT "
            r"sample_checksum=[-0-9.]+ sample_row0_col0=[-0-9.]+ "
            r"sample_last=[-0-9.]+",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_END "
            r"scenario=flash_attention_h100_1sm_compare "
            r"compare_scope=full_flash_attention "
            r"softmax_scope=cpu_helper status=PASS",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_PERF_SOURCE "
            r"source_line=profile_and_MPU_SUMMARY "
            r"cycles_field=profile_stage_span_cycles "
            r"latency_field=profile_compute_exec_cycles "
            r"macs_field=mpu_summary_total_macs "
            r"busy_field=mpu_summary_busy "
            r"idle_field=profile_stage_idle "
            r"spm_wait_field=mpu_summary_spm_wait",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_PERF_SUMMARY "
            r"scenario=flash_attention_h100_1sm_compare "
            r"compare_cycles=[0-9]+ compare_latency=[0-9]+ "
            r"compare_macs=[0-9]+ compare_busy=[0-9]+ "
            r"compare_idle=[0-9]+ compare_spm_wait=[0-9]+ "
            r"compare_scope=full_flash_attention "
            r"softmax_scope=cpu_helper "
            r"excluded_from_compare_latency=true status=PASS",
            r"FLASH_ATTENTION_H100_1SM_COMPARE_PASS",
            rf"MPU_SUMMARY scenario={SCENARIO} .* compute_cycles=[0-9]+ "
            r"cmd_cycles=[0-9]+ spm_wait=[0-9]+ macs=[0-9]+ "
            r"busy=[0-9]+ idle=[0-9]+ .*",
            r"MPU_EXIT_CODE=0",
            make_profile_artifact_verifier(__file__),
            GeneratePerformanceSummary(),
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
