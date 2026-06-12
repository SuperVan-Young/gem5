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

SCENARIO = "flash_attention_v2_operator_perf_q256_kv1024_d128"
TESTCASE_ROOT = Path(__file__).resolve().parent
PROFILE_DIR = TESTCASE_ROOT / "profile"
PROFILE_LOG = PROFILE_DIR / "flashAttentionV2.npu_profile.log"
PROFILE_JSON = PROFILE_DIR / "flashAttentionV2.npu_profile.json"
PROFILE_HTML = PROFILE_DIR / "flashAttentionV2.npu_profile.html"
STDOUT_ARTIFACT = PROFILE_DIR / "flashAttentionV2.simout.txt"
PERF_SUMMARY = PROFILE_DIR / "flashAttentionV2.performance_summary.json"
PERF_SUMMARIZER = TESTCASE_ROOT / "summarize_perf.py"
binary = resolve_binary_path(
    __file__, "flash_attention_v2_operator_perf_q256_kv1024_d128_riscv"
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

        try:
            summary = json.loads(PERF_SUMMARY.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            test_util.fail("Invalid JSON in %s: %s", PERF_SUMMARY, exc)

        if summary["shape"]["total_matmul_flops"] != 134217728:
            test_util.fail("Unexpected total matmul FLOPs in %s", PERF_SUMMARY)
        if summary["profile"]["qk_fast_matmul"]["macro_count"] != 1:
            test_util.fail("QK fastMatmul should have one macro")
        if summary["profile"]["pv_fast_matmul"]["macro_count"] != 1:
            test_util.fail("PV fastMatmul should have one macro")
        if summary["profile"]["fast_online_softmax"]["macro_count"] != 14:
            test_util.fail("fastOnlineSoftmax should have 14 macros")
        if summary["profile"]["operator_total"]["macro_count"] != 16:
            test_util.fail("FlashAttentionV2 operator should have 16 macros")
        if summary["mpu_summary"]["cmds"] != 2:
            test_util.fail("Expected two MPU fastMatmul commands")
        if summary["vpu_summary"]["cmds"] != 14:
            test_util.fail("Expected 14 VPU softmax commands")
        if summary["vpu_summary"]["max_active_uops"] <= 1:
            test_util.fail("Expected pipelined VPU uops")
        if summary["vpu_summary"]["executes"] > 800:
            test_util.fail("Expected grouped VPU dlen uops")
        if summary["performance"]["effective_tops"] <= 0.0:
            test_util.fail("Expected positive effective TOPS")
        if summary["performance"]["utilization"] <= 0.0:
            test_util.fail("Expected positive utilization")


register_npu_test(
    NpuRunnerSpec(
        name="flash_attention_v2_operator_perf_q256_kv1024_d128",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(binary, "--scenario", SCENARIO)
        ),
        gem5_args=make_profile_gem5_args(
            __file__, artifact_name="flashAttentionV2.npu_profile.log"
        ),
        verifier_specs=(
            rf"FLASH_ATTENTION_V2_SCENARIO={SCENARIO}",
            r"FLASH_ATTENTION_V2_SHAPE q=256 kv=1024 d=128 "
            r"tile_m=128 tile_n=128 tile_k=128 matmul_count=2 "
            r"single_matmul_flops=67108864 total_matmul_flops=134217728",
            r"FLASH_ATTENTION_V2_QK_FAST_MATMUL cmds=1 template_builds=1 "
            r"sync_indicator=129 status=PASS",
            r"FLASH_ATTENTION_V2_FAST_ONLINE_SOFTMAX cmds=14 "
            r"sync_indicator=145 status=PASS",
            r"FLASH_ATTENTION_V2_PV_FAST_MATMUL cmds=1 template_builds=1 "
            r"sync_indicator=130 status=PASS",
            r"FLASH_ATTENTION_V2_PROFILE_QK total_span_cycles=[0-9]+ "
            r"busy_cycles=[0-9]+ macro_count=1 fused_matmul=1 status=PASS",
            r"FLASH_ATTENTION_V2_PROFILE_SOFTMAX total_span_cycles=[0-9]+ "
            r"busy_cycles=[0-9]+ macro_count=14 reduce_max=1 add=3 sub=3 "
            r"mul=1 div=1 scale=1 abs=1 reduce_sum=1 exp=2 status=PASS",
            r"FLASH_ATTENTION_V2_PROFILE_PV total_span_cycles=[0-9]+ "
            r"busy_cycles=[0-9]+ macro_count=1 fused_matmul=1 status=PASS",
            r"FLASH_ATTENTION_V2_PROFILE_TOTAL total_span_cycles=[0-9]+ "
            r"busy_cycles=[0-9]+ macro_count=16 status=PASS",
            r"FLASH_ATTENTION_V2_PASS",
            r"FLASH_ATTENTION_V2_MPU_SUMMARY cmds=2 "
            r"compute_cycles=[0-9]+ drain_cycles=[0-9]+ "
            r"output_ready_cycles=[0-9]+ cmd_cycles=[0-9]+ "
            r"spm_wait=[0-9]+ macs=[0-9]+ busy=[0-9]+ idle=[0-9]+",
            r"FLASH_ATTENTION_V2_VPU_SUMMARY cmds=14 queue_occupancy=0 "
            r"issue_busy=False executes=[0-9]+ "
            r"max_active_uops=([2-9]|[1-9][0-9]+)",
            r"FLASH_ATTENTION_V2_CONFIG_PASS",
            make_profile_artifact_verifier(
                __file__,
                raw_artifact_name="flashAttentionV2.npu_profile.log",
                parsed_artifact_name="flashAttentionV2.npu_profile.json",
                html_artifact_name="flashAttentionV2.npu_profile.html",
            ),
            GeneratePerformanceSummary(),
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
