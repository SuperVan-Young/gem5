# Copyright (c) 2026
# All rights reserved.

"""Reusable harness helpers for NPU testcase runner files.

This module keeps testcase harness files declarative by centralizing:

- stable regex verifier construction,
- testcase-root resolution and MakeTarget/MakeFixture creation,
- config/binary path resolution from a testcase entry location,
- single-test and multi-scenario gem5_verify_config registration.

The helpers intentionally stay thin. They do not build gem5 systems and do not
encode testcase-specific simulation semantics.
"""

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from testlib import test_util
from testlib.configuration import constants

from gem5 import verifier
from gem5.fixture import (
    MakeFixture,
    MakeTarget,
)
from gem5.suite import gem5_verify_config

NPU_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
NPU_PROFILE_PARSE_TOOL = NPU_TOOLS_DIR / "parse_npu_profile.py"
NPU_PROFILE_RENDER_TOOL = NPU_TOOLS_DIR / "render_npu_profile.py"


def resolve_testcase_root(reference_file):
    return Path(reference_file).resolve().parent


def testcase_dir(reference_file):
    return resolve_testcase_root(reference_file)


def resolve_config_path(reference_file, config_filename="config.py"):
    return resolve_testcase_root(reference_file) / config_filename


def resolve_binary_path(reference_file, binary_name):
    return resolve_testcase_root(reference_file) / "bin" / binary_name


def resolve_testcase_profile_dir(reference_file):
    profile_dir = resolve_testcase_root(reference_file) / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def resolve_testcase_profile_path(reference_file, artifact_name=None):
    testcase_root = resolve_testcase_root(reference_file)
    if artifact_name is None:
        artifact_name = f"{testcase_root.name}.npu_profile.log"
    return resolve_testcase_profile_dir(reference_file) / artifact_name


def resolve_testcase_profile_json_path(reference_file, artifact_name=None):
    testcase_root = resolve_testcase_root(reference_file)
    if artifact_name is None:
        artifact_name = f"{testcase_root.name}.npu_profile.json"
    return resolve_testcase_profile_dir(reference_file) / artifact_name


def resolve_testcase_profile_html_path(reference_file, artifact_name=None):
    testcase_root = resolve_testcase_root(reference_file)
    if artifact_name is None:
        artifact_name = f"{testcase_root.name}.npu_profile.html"
    return resolve_testcase_profile_dir(reference_file) / artifact_name


def make_profile_gem5_args(
    reference_file, artifact_name=None, debug_flags=("NPUProfile",)
):
    gem5_args = []
    if debug_flags:
        gem5_args.append(f"--debug-flags={','.join(debug_flags)}")
    gem5_args.append(
        f"--debug-file={resolve_testcase_profile_path(reference_file, artifact_name)}"
    )
    return tuple(gem5_args)


class GenerateProfileArtifacts(verifier.Verifier):
    def __init__(self, raw_profile, parsed_profile, html_profile):
        super().__init__()
        self.raw_profile = Path(raw_profile)
        self.parsed_profile = Path(parsed_profile)
        self.html_profile = Path(html_profile)

    def _run_tool(self, argv, tool_name):
        result = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            test_util.fail(
                "%s failed for %s:\nstdout:\n%s\nstderr:\n%s",
                tool_name,
                self.raw_profile,
                result.stdout,
                result.stderr,
            )

    def test(self, params):
        del params
        if not self.raw_profile.is_file():
            test_util.fail("Missing raw profile log %s", self.raw_profile)
        if self.raw_profile.stat().st_size == 0:
            test_util.fail("Empty raw profile log %s", self.raw_profile)

        self._run_tool(
            [
                sys.executable,
                str(NPU_PROFILE_PARSE_TOOL),
                "--input",
                str(self.raw_profile),
                "--output",
                str(self.parsed_profile),
            ],
            "parse_npu_profile.py",
        )
        self._run_tool(
            [
                sys.executable,
                str(NPU_PROFILE_RENDER_TOOL),
                "--input",
                str(self.parsed_profile),
                "--output",
                str(self.html_profile),
            ],
            "render_npu_profile.py",
        )


def make_profile_artifact_verifier(
    reference_file,
    raw_artifact_name=None,
    parsed_artifact_name=None,
    html_artifact_name=None,
):
    return GenerateProfileArtifacts(
        resolve_testcase_profile_path(reference_file, raw_artifact_name),
        resolve_testcase_profile_json_path(reference_file, parsed_artifact_name),
        resolve_testcase_profile_html_path(reference_file, html_artifact_name),
    )


def make_testcase_build_fixture(reference_file, target="all", source_dir=None):
    testcase_root = resolve_testcase_root(reference_file)
    if source_dir is None:
        build_root = testcase_root
    else:
        build_root = testcase_root / source_dir
    return MakeTarget(target, make_fixture=MakeFixture(str(build_root)))


def make_binary_config_args(binary, *extra_args):
    return ["--binary", str(binary), *[str(arg) for arg in extra_args]]


def make_named_regex_verifier(
    name, regex, *, match_stderr=True, match_stdout=True
):
    if isinstance(regex, (str, bytes)):
        regex = re.compile(regex)
    verifier_cls = type(name, (verifier.MatchRegex,), {})
    return verifier_cls(
        regex,
        match_stderr=match_stderr,
        match_stdout=match_stdout,
    )


def _as_sequence(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def build_verifiers(
    name_prefix,
    verifier_specs,
    *,
    match_stderr=True,
    match_stdout=True,
):
    if verifier_specs is None:
        return []

    if isinstance(verifier_specs, verifier.Verifier):
        verifier_specs = (verifier_specs,)
    elif isinstance(verifier_specs, (str, bytes)) or hasattr(
        verifier_specs, "pattern"
    ):
        verifier_specs = (verifier_specs,)
    else:
        verifier_specs = tuple(verifier_specs)

    built = []
    for index, spec in enumerate(verifier_specs, start=1):
        if isinstance(spec, verifier.Verifier):
            built.append(spec)
            continue

        verifier_name = f"{name_prefix}Verifier{index}"
        built.append(
            make_named_regex_verifier(
                verifier_name,
                spec,
                match_stderr=match_stderr,
                match_stdout=match_stdout,
            )
        )

    return built


@dataclass(frozen=True)
class NpuRunnerSpec:
    name: str
    config: Any
    config_args: tuple[Any, ...] = ()
    gem5_args: tuple[Any, ...] = ()
    verifier_specs: Any = ()
    fixtures: tuple[Any, ...] = ()
    valid_isas: tuple[Any, ...] = (constants.riscv_tag,)
    valid_variants: tuple[Any, ...] = constants.supported_variants
    length: Any = constants.quick_tag
    valid_hosts: tuple[Any, ...] = constants.supported_hosts
    protocol: Any = None
    uses_kvm: bool = False


@dataclass(frozen=True)
class NpuScenarioSpec:
    suffix: str
    config_args: tuple[Any, ...] = ()
    verifier_specs: Any = None
    gem5_args: tuple[Any, ...] | None = None


def register_npu_test(spec: NpuRunnerSpec):
    return gem5_verify_config(
        name=spec.name,
        verifiers=build_verifiers(spec.name, spec.verifier_specs),
        config=str(spec.config),
        config_args=[str(arg) for arg in spec.config_args],
        gem5_args=[str(arg) for arg in spec.gem5_args],
        fixtures=list(spec.fixtures),
        valid_isas=spec.valid_isas,
        valid_variants=spec.valid_variants,
        length=spec.length,
        valid_hosts=spec.valid_hosts,
        protocol=spec.protocol,
        uses_kvm=spec.uses_kvm,
    )


def register_npu_scenarios(base_spec: NpuRunnerSpec, scenarios):
    registered = []
    for scenario in scenarios:
        registered.extend(
            register_npu_test(
                NpuRunnerSpec(
                    name=f"{base_spec.name}_{scenario.suffix}",
                    config=base_spec.config,
                    config_args=(
                        *_as_sequence(base_spec.config_args),
                        *scenario.config_args,
                    ),
                    gem5_args=(
                        scenario.gem5_args
                        if scenario.gem5_args is not None
                        else base_spec.gem5_args
                    ),
                    verifier_specs=(
                        scenario.verifier_specs
                        if scenario.verifier_specs is not None
                        else base_spec.verifier_specs
                    ),
                    fixtures=base_spec.fixtures,
                    valid_isas=base_spec.valid_isas,
                    valid_variants=base_spec.valid_variants,
                    length=base_spec.length,
                    valid_hosts=base_spec.valid_hosts,
                    protocol=base_spec.protocol,
                    uses_kvm=base_spec.uses_kvm,
                )
            )
        )
    return registered
