#!/usr/bin/env python3
#
# Copyright (c) 2026
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer; redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution; neither the name of the copyright holders nor the
# names of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import re
import sys
from pathlib import Path
from unittest.mock import patch
import unittest

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "ext"))
sys.path.insert(0, str(repo_root / "tests"))
sys.path.insert(0, str(repo_root / "tests" / "gem5" / "npu" / "configs"))

from runner_common import (  # noqa: E402
    NpuRunnerSpec,
    NpuScenarioSpec,
    build_verifiers,
    make_binary_config_args,
    make_testcase_build_fixture,
    make_named_regex_verifier,
    register_npu_scenarios,
    register_npu_test,
    resolve_binary_path,
    resolve_config_path,
    resolve_testcase_root,
    testcase_dir,
)


class NpuRunnerCommonTestCase(unittest.TestCase):
    def test_path_helpers(self):
        reference = "/tmp/example/test.py"
        self.assertEqual(
            resolve_testcase_root(reference),
            Path("/tmp/example"),
        )
        self.assertEqual(testcase_dir(reference), Path("/tmp/example"))
        self.assertEqual(
            resolve_config_path(reference),
            Path("/tmp/example/config.py"),
        )
        self.assertEqual(
            resolve_binary_path(reference, "seu_mmio_riscv"),
            Path("/tmp/example/bin/seu_mmio_riscv"),
        )

    def test_make_binary_config_args(self):
        self.assertEqual(
            make_binary_config_args(Path("/tmp/bin/test"), "--rounds", 4),
            ["--binary", "/tmp/bin/test", "--rounds", "4"],
        )

    def test_make_testcase_build_fixture(self):
        with patch("runner_common.MakeFixture") as mock_make_fixture, patch(
            "runner_common.MakeTarget"
        ) as mock_make_target:
            make_testcase_build_fixture("/tmp/example/test.py")

        mock_make_fixture.assert_called_once_with("/tmp/example")
        mock_make_target.assert_called_once_with(
            "all",
            make_fixture=mock_make_fixture.return_value,
        )

    def test_make_testcase_build_fixture_allows_source_subdir(self):
        with patch("runner_common.MakeFixture") as mock_make_fixture, patch(
            "runner_common.MakeTarget"
        ) as mock_make_target:
            make_testcase_build_fixture(
                "/tmp/example/test.py",
                source_dir="src",
            )

        mock_make_fixture.assert_called_once_with("/tmp/example/src")
        mock_make_target.assert_called_once_with(
            "all",
            make_fixture=mock_make_fixture.return_value,
        )

    def test_make_named_regex_verifier(self):
        regex = re.compile(r"PASS")
        instance = make_named_regex_verifier("MyVerifier", regex)

        self.assertEqual(instance.__class__.__name__, "MyVerifier")
        self.assertEqual([item.pattern for item in instance.regex], ["PASS"])
        self.assertEqual(
            instance.filenames,
            ["simout.txt", "simerr.txt"],
        )

    def test_build_verifiers_mixes_regex_and_instances(self):
        direct = make_named_regex_verifier("DirectVerifier", re.compile(r"X"))
        built = build_verifiers(
            "Runner",
            (re.compile(r"A"), "B", direct),
        )

        self.assertEqual([v.__class__.__name__ for v in built], [
            "RunnerVerifier1",
            "RunnerVerifier2",
            "DirectVerifier",
        ])

    def test_register_npu_test_normalizes_inputs(self):
        spec = NpuRunnerSpec(
            name="seu_basic",
            config=Path("/tmp/testcase/config.py"),
            config_args=(Path("/tmp/testcase/bin/seu"), "--binary", 1),
            gem5_args=("--debug-flags=SpecializedExecutionUnit",),
            verifier_specs=re.compile(r"SEU_TEST_PASS"),
            fixtures=("fixture",),
        )

        with patch("runner_common.gem5_verify_config") as mock_register:
            mock_register.return_value = [object()]
            register_npu_test(spec)

        mock_register.assert_called_once()
        kwargs = mock_register.call_args.kwargs
        self.assertEqual(kwargs["name"], "seu_basic")
        self.assertEqual(kwargs["config"], "/tmp/testcase/config.py")
        self.assertEqual(
            kwargs["config_args"],
            ["/tmp/testcase/bin/seu", "--binary", "1"],
        )
        self.assertEqual(
            kwargs["gem5_args"],
            ["--debug-flags=SpecializedExecutionUnit"],
        )
        self.assertEqual(kwargs["fixtures"], ["fixture"])
        self.assertEqual(
            kwargs["verifiers"][0].__class__.__name__,
            "seu_basicVerifier1",
        )

    def test_register_npu_scenarios_merges_overrides(self):
        base_spec = NpuRunnerSpec(
            name="system_pipeline",
            config="/tmp/testcase/config.py",
            config_args=("--binary", "/tmp/testcase/bin/system"),
            gem5_args=("--debug-flags=Base",),
            verifier_specs=re.compile(r"BASE"),
            fixtures=("fixture",),
        )
        scenarios = [
            NpuScenarioSpec(
                suffix="spm_only",
                config_args=("--scenario", "spm_only"),
                verifier_specs=re.compile(r"SPM_ONLY"),
            ),
            NpuScenarioSpec(
                suffix="copy_back",
                config_args=("--scenario", "copy_back"),
                gem5_args=("--debug-flags=CopyBack",),
            ),
        ]

        with patch("runner_common.gem5_verify_config") as mock_register:
            mock_register.return_value = [object()]
            register_npu_scenarios(base_spec, scenarios)

        self.assertEqual(mock_register.call_count, 2)
        first = mock_register.call_args_list[0].kwargs
        second = mock_register.call_args_list[1].kwargs
        self.assertEqual(first["name"], "system_pipeline_spm_only")
        self.assertEqual(
            first["config_args"],
            ["--binary", "/tmp/testcase/bin/system", "--scenario", "spm_only"],
        )
        self.assertEqual(
            first["verifiers"][0].__class__.__name__,
            "system_pipeline_spm_onlyVerifier1",
        )
        self.assertEqual(second["name"], "system_pipeline_copy_back")
        self.assertEqual(
            second["gem5_args"],
            ["--debug-flags=CopyBack"],
        )
        self.assertEqual(
            second["verifiers"][0].__class__.__name__,
            "system_pipeline_copy_backVerifier1",
        )


if __name__ == "__main__":
    unittest.main()
