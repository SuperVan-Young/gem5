# Copyright (c) 2026
# All rights reserved.

import json
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
binary = resolve_binary_path(
    __file__, "flash_attention_matmul_q256_kv1024_d128_riscv"
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

        if summary["baseline"]["build_cmd_calls"] != 112:
            test_util.fail(
                "Unexpected baseline build count in %s", PERF_SUMMARY
            )
        if summary["fast"]["template_builds"] != 7:
            test_util.fail(
                "Unexpected fast template count in %s", PERF_SUMMARY
            )
        if (
            summary["compare"]["baseline_checksum"]
            != summary["compare"]["fast_checksum"]
        ):
            test_util.fail("Output checksum mismatch in %s", PERF_SUMMARY)


register_npu_test(
    NpuRunnerSpec(
        name="flash_attention_matmul_q256_kv1024_d128",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(binary, "--scenario", SCENARIO)
        ),
        gem5_args=make_profile_gem5_args(__file__),
        verifier_specs=(
            rf"FLASH_ATTENTION_MATMUL_SCENARIO={SCENARIO}",
            r"FLASH_ATTENTION_MATMUL_SHAPE "
            r"m=256 n=1024 k=128 tile_m=128 tile_n=128 tile_k=128 "
            r"total_flops=67108864",
            r"FLASH_ATTENTION_MATMUL_BASELINE cmds=112 build_cmd_calls=112 "
            r"sync_indicator=64 status=PASS",
            r"FLASH_ATTENTION_MATMUL_FAST cmds=112 template_builds=7 "
            r"sync_indicator=128 status=PASS",
            r"FLASH_ATTENTION_MATMUL_COMPARE build_call_reduction=105 "
            r"baseline_checksum=-?[0-9]+ fast_checksum=-?[0-9]+ "
            r"ref_checksum=-?[0-9]+ status=PASS",
            r"FLASH_ATTENTION_MATMUL_PROFILE_BASELINE "
            r"total_span_cycles=[0-9]+ busy_cycles=[0-9]+ "
            r"compute_cycles=[0-9]+ macro_count=112 "
            r"mvin=32 load=32 compute=16 drain=16 mvout=16 status=PASS",
            r"FLASH_ATTENTION_MATMUL_PROFILE_FAST "
            r"total_span_cycles=[0-9]+ busy_cycles=[0-9]+ "
            r"compute_cycles=[0-9]+ macro_count=112 "
            r"mvin=32 load=32 compute=16 drain=16 mvout=16 status=PASS",
            r"FLASH_ATTENTION_MATMUL_PROFILE_COMPARE "
            r"baseline_total_span_cycles=[0-9]+ "
            r"fast_total_span_cycles=[0-9]+ "
            r"baseline_busy_cycles=[0-9]+ fast_busy_cycles=[0-9]+ "
            r"baseline_compute_cycles=[0-9]+ fast_compute_cycles=[0-9]+ "
            r"status=PASS",
            r"FLASH_ATTENTION_MATMUL_PASS",
            rf"MPU_SUMMARY scenario={SCENARIO} cmds=224 "
            r"compute_cycles=[0-9]+ cmd_cycles=[0-9]+ spm_wait=[0-9]+ "
            r"macs=67108864 busy=[0-9]+ idle=[0-9]+",
            r"MPU_EXIT_CODE=0",
            make_profile_artifact_verifier(__file__),
            GeneratePerformanceSummary(),
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
