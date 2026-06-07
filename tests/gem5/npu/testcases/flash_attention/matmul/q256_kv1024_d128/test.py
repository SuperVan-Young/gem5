# Copyright (c) 2026
# All rights reserved.

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "configs"))

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

SCENARIO = "flash_attention_matmul_q256_kv1024_d128"
TESTCASE_ROOT = Path(__file__).resolve().parent
PROFILE_DIR = TESTCASE_ROOT / "profile"
PROFILE_LOG = PROFILE_DIR / "q256_kv1024_d128.npu_profile.log"
PROFILE_JSON = PROFILE_DIR / "q256_kv1024_d128.npu_profile.json"
PROFILE_HTML = PROFILE_DIR / "q256_kv1024_d128.npu_profile.html"
STDOUT_ARTIFACT = PROFILE_DIR / "q256_kv1024_d128.simout.txt"
PERF_SUMMARY = PROFILE_DIR / "q256_kv1024_d128.performance_summary.json"
PERF_SUMMARIZER = TESTCASE_ROOT / "summarize_perf.py"
FAST_MATMUL_HEADER = (
    TESTCASE_ROOT.parents[3] / "utils" / "primitive" / "fastMatmul.hh"
)
binary = resolve_binary_path(
    __file__, "flash_attention_matmul_q256_kv1024_d128_riscv"
)


def make_matmul_profile_gem5_args():
    args = list(make_profile_gem5_args(__file__))
    for idx, arg in enumerate(args):
        if arg.startswith("--debug-flags="):
            args[idx] = arg + ",NPULaunchProfile"
            return tuple(args)
    return tuple(args + ["--debug-flags=NPUProfile,NPULaunchProfile"])


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

        if summary["baseline"]["build_cmd_calls"] != 112:
            test_util.fail(
                "Unexpected baseline build count in %s", PERF_SUMMARY
            )
        if summary["fast"]["template_builds"] != 5:
            test_util.fail(
                "Unexpected fast template count in %s", PERF_SUMMARY
            )
        if summary["baseline"]["profile"]["contiguous_load_macro_count"] == 0:
            test_util.fail(
                "Baseline phase should contain contiguous load macros in %s",
                PERF_SUMMARY,
            )
        if summary["baseline"]["profile"]["contiguous_load_max_uops"] != 1:
            test_util.fail(
                "Baseline contiguous loads should issue one uop in %s",
                PERF_SUMMARY,
            )
        if summary["fast"]["profile"]["contiguous_load_macro_count"] == 0:
            test_util.fail(
                "Fast phase should contain contiguous load macros in %s",
                PERF_SUMMARY,
            )
        if summary["fast"]["profile"]["contiguous_load_max_uops"] != 1:
            test_util.fail(
                "Fast contiguous loads should issue one uop in %s",
                PERF_SUMMARY,
            )
        if summary["fast"]["profile"]["fused_compute_macro_count"] != 16:
            test_util.fail(
                "Expected 16 fast fused compute macros in %s", PERF_SUMMARY
            )
        if (
            summary["compare"]["baseline_checksum"]
            != summary["compare"]["fast_checksum"]
        ):
            test_util.fail("Output checksum mismatch in %s", PERF_SUMMARY)
        if "fast_issue_gaps" not in summary:
            test_util.fail("Missing fast_issue_gaps in %s", PERF_SUMMARY)
        if summary["fast_issue_gaps"]["macro_count"] != 80:
            test_util.fail(
                "Expected 80 fast macro events, got %s",
                summary["fast_issue_gaps"]["macro_count"],
            )
        if "launch_profile" not in summary:
            test_util.fail("Missing launch_profile in %s", PERF_SUMMARY)
        if summary["launch_profile"]["count"] != 192:
            test_util.fail(
                "Expected 192 launch profile events, got %s",
                summary["launch_profile"]["count"],
            )
        if "launch_request_gaps" not in summary:
            test_util.fail("Missing launch_request_gaps in %s", PERF_SUMMARY)
        if summary["launch_request_gaps"]["count"] != 80:
            test_util.fail(
                "Expected 80 fast launch request events, got %s",
                summary["launch_request_gaps"]["count"],
            )
        if summary["launch_request_gaps"]["gap_count"] != 79:
            test_util.fail(
                "Expected 79 fast launch request gaps, got %s",
                summary["launch_request_gaps"]["gap_count"],
            )
        if summary["fast_effective_tops"] <= 0.0:
            test_util.fail(
                "Expected positive fast_effective_tops in %s", PERF_SUMMARY
            )
        if summary["fast_utilization_16tops"] <= 0.0:
            test_util.fail(
                "Expected positive fast_utilization_16tops in %s",
                PERF_SUMMARY,
            )


class VerifyFastMatmulLaunchPath(verifier.Verifier):
    def __init__(self):
        super().__init__()

    def test(self, params):
        source = FAST_MATMUL_HEADER.read_text(encoding="utf-8")
        if re.search(r"\.launchCmdAt\s*\(", source):
            test_util.fail(
                "fastMatmul should call the flat raw-pairs launch path, "
                "not NpuCmd::launchCmdAt()"
            )
        if "npuCmdLaunchRawPairsInsn(" not in source:
            test_util.fail(
                "fastMatmul should prepare raw command pairs and launch "
                "through npuCmdLaunchRawPairsInsn()"
            )


class VerifyFastMatmulFusedComputePath(verifier.Verifier):
    def __init__(self):
        super().__init__()

    def test(self, params):
        source = FAST_MATMUL_HEADER.read_text(encoding="utf-8")
        if "MPU_OP_COMPUTE_FUSED" not in source:
            test_util.fail("fastMatmul should use fused compute opcode")
        forbidden = (
            "templates.load_a,",
            "templates.load_b,",
            "templates.compute,",
        )
        for pattern in forbidden:
            if pattern in source:
                test_util.fail(
                    "fastMatmul should not launch separate %s", pattern
                )


register_npu_test(
    NpuRunnerSpec(
        name="flash_attention_matmul_q256_kv1024_d128",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(binary, "--scenario", SCENARIO)
        ),
        gem5_args=make_matmul_profile_gem5_args(),
        verifier_specs=(
            rf"FLASH_ATTENTION_MATMUL_SCENARIO={SCENARIO}",
            r"FLASH_ATTENTION_MATMUL_SHAPE "
            r"m=256 n=1024 k=128 tile_m=128 tile_n=128 tile_k=128 "
            r"total_flops=67108864",
            r"FLASH_ATTENTION_MATMUL_BASELINE cmds=112 build_cmd_calls=112 "
            r"sync_indicator=64 status=PASS",
            r"FLASH_ATTENTION_MATMUL_FAST cmds=80 template_builds=5 "
            r"sync_indicator=128 status=PASS",
            r"FLASH_ATTENTION_MATMUL_COMPARE build_call_reduction=107 "
            r"baseline_checksum=-?[0-9]+ fast_checksum=-?[0-9]+ "
            r"ref_checksum=-?[0-9]+ status=PASS",
            r"FLASH_ATTENTION_MATMUL_PROFILE_BASELINE "
            r"total_span_cycles=[0-9]+ busy_cycles=[0-9]+ "
            r"compute_cycles=[0-9]+ macro_count=112 "
            r"mvin=32 load=32 compute=16 drain=16 mvout=16 status=PASS",
            r"FLASH_ATTENTION_MATMUL_PROFILE_FAST "
            r"total_span_cycles=[0-9]+ busy_cycles=[0-9]+ "
            r"compute_cycles=[0-9]+ macro_count=80 "
            r"mvin=32 load=0 compute=16 drain=16 mvout=16 status=PASS",
            r"FLASH_ATTENTION_MATMUL_PROFILE_COMPARE "
            r"baseline_total_span_cycles=[0-9]+ "
            r"fast_total_span_cycles=[0-9]+ "
            r"baseline_busy_cycles=[0-9]+ fast_busy_cycles=[0-9]+ "
            r"baseline_compute_cycles=[0-9]+ fast_compute_cycles=[0-9]+ "
            r"status=PASS",
            r"FLASH_ATTENTION_MATMUL_PASS",
            rf"MPU_SUMMARY scenario={SCENARIO} cmds=192 "
            r"compute_cycles=[0-9]+ drain_cycles=[0-9]+ "
            r"output_ready_cycles=[0-9]+ cmd_cycles=[0-9]+ "
            r"spm_wait=[0-9]+ "
            r"macs=67108864 busy=[0-9]+ idle=[0-9]+",
            r"MPU_EXIT_CODE=0",
            make_profile_artifact_verifier(__file__),
            GeneratePerformanceSummary(),
            VerifyFastMatmulLaunchPath(),
            VerifyFastMatmulFusedComputePath(),
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
