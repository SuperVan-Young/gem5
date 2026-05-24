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

SCENARIO = "flash_attention_fast_online_softmax_256x1024"
TESTCASE_ROOT = Path(__file__).resolve().parent
PROFILE_DIR = TESTCASE_ROOT / "profile"
PROFILE_LOG = PROFILE_DIR / "fastOnlineSoftmax.npu_profile.log"
PROFILE_JSON = PROFILE_DIR / "fastOnlineSoftmax.npu_profile.json"
PROFILE_HTML = PROFILE_DIR / "fastOnlineSoftmax.npu_profile.html"
STDOUT_ARTIFACT = PROFILE_DIR / "fastOnlineSoftmax.simout.txt"
PERF_SUMMARY = PROFILE_DIR / "fastOnlineSoftmax.performance_summary.json"
PERF_SUMMARIZER = TESTCASE_ROOT / "summarize_perf.py"
binary = resolve_binary_path(
    __file__, "flash_attention_fast_online_softmax_256x1024_riscv"
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

        if summary["cmd"]["template_builds"] != 14:
            test_util.fail(
                "Unexpected template build count in %s", PERF_SUMMARY
            )
        if summary["cmd"]["launched_cmds"] != 14:
            test_util.fail("Unexpected launched cmd count in %s", PERF_SUMMARY)
        if summary["profile"]["macro_count"] != 14:
            test_util.fail("Unexpected macro count in %s", PERF_SUMMARY)
        if summary["profile"]["opcode_counts"]["sub"] != 3:
            test_util.fail("Unexpected sub count in %s", PERF_SUMMARY)
        if summary["profile"]["opcode_counts"]["exp"] != 2:
            test_util.fail("Unexpected exp count in %s", PERF_SUMMARY)


register_npu_test(
    NpuRunnerSpec(
        name="flash_attention_fast_online_softmax_256x1024",
        config=resolve_config_path(__file__),
        config_args=tuple(
            make_binary_config_args(binary, "--scenario", SCENARIO)
        ),
        gem5_args=make_profile_gem5_args(
            __file__, artifact_name="fastOnlineSoftmax.npu_profile.log"
        ),
        verifier_specs=(
            rf"FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_SCENARIO={SCENARIO}",
            r"FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_SHAPE "
            r"rows=256 cols=1024 total_elements=262144 scratch_slots=[0-9]+",
            r"FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_CMD "
            r"templates=14 launched=14 slices=1 "
            r"sync_indicator=145 status=PASS",
            r"FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_ACCURACY "
            r"max_abs_m=[0-9.]+ max_abs_l=[0-9.]+ max_abs_p=[0-9.]+ "
            r"sample_checksum=[-0-9.]+ status=PASS",
            r"FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PROFILE "
            r"total_span_cycles=[0-9]+ busy_cycles=[0-9]+ macro_count=14 "
            r"reduce_max=1 add=3 sub=3 mul=1 div=1 scale=1 abs=1 "
            r"reduce_sum=1 exp=2 status=PASS",
            r"FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PASS",
            rf"VPU_SUMMARY scenario={SCENARIO} cmds=14 queue_occupancy=0 "
            r"issue_busy=False executes=14 max_active_uops=[0-9]+",
            r"VPU_EXIT_CODE=0",
            r"FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_CONFIG_PASS",
            make_profile_artifact_verifier(
                __file__,
                raw_artifact_name="fastOnlineSoftmax.npu_profile.log",
                parsed_artifact_name="fastOnlineSoftmax.npu_profile.json",
                html_artifact_name="fastOnlineSoftmax.npu_profile.html",
            ),
            GeneratePerformanceSummary(),
        ),
        fixtures=(make_testcase_build_fixture(__file__),),
    )
)
