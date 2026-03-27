# Copyright (c) 2026
# All rights reserved.

import re
import subprocess
import sys
from pathlib import Path

from testlib import *
from testlib.helper import log_call

from gem5.fixture import (
    Gem5Fixture,
    MakeFixture,
    MakeTarget,
    TempdirFixture,
)

dma_dir = Path(__file__).resolve().parent
binary = dma_dir / "bin" / "dma_proto_riscv"
dma_debug_flags = "--debug-flags=DMA"
dma_max_ticks = "5000000000"
config_path = joinpath(
    config.base_dir,
    "tests",
    "gem5",
    "npu",
    "dma",
    "configs",
    "dma_proto.py",
)
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(dma_dir / "src")),
)
dma_scenarios = (
    ("dma_basic_dram_to_spm", "basic_dram_to_spm"),
    ("dma_basic_spm_to_dram", "basic_spm_to_dram"),
    ("dma_hwc_to_blocked", "hwc_to_blocked"),
    ("dma_blocked_to_blocked", "blocked_to_blocked"),
    ("dma_buffer_size_forces_batching", "buffer_size_forces_batching"),
    ("dma_sync_completion", "sync_completion"),
)


def add_dma_test(name, scenario):
    gem5_verify_config(
        name=name,
        verifiers=[
            verifier.MatchRegex(
                re.compile(rf"DMA_SCENARIO_PASS={re.escape(scenario)}")
            )
        ],
        config=config_path,
        config_args=[
            "--binary",
            str(binary),
            "--scenario",
            scenario,
            "--max-ticks",
            dma_max_ticks,
        ],
        gem5_args=[dma_debug_flags],
        valid_isas=(constants.riscv_tag,),
        valid_hosts=constants.supported_hosts,
        length=constants.quick_tag,
        fixtures=[build_fixture],
    )


for test_name, scenario_name in dma_scenarios:
    add_dma_test(test_name, scenario_name)


def run_expected_invalid_address(params):
    fixtures = params.fixtures
    tempdir = fixtures[constants.tempdir_fixture_name].path
    gem5 = fixtures[constants.gem5_binary_fixture_name].path
    command = [
        gem5,
        "-d",
        tempdir,
        "-re",
        "--silent-redirect",
        dma_debug_flags,
        config_path,
        "--binary",
        str(binary),
        "--scenario",
        "invalid_address",
        "--max-ticks",
        dma_max_ticks,
    ]

    try:
        log_call(
            params.log,
            command,
            time=params.time,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError:
        return

    raise AssertionError(
        "Expected invalid_address scenario to terminate gem5 with a panic"
    )


for host in constants.supported_hosts:
    for opt in constants.supported_variants:
        for isa in (constants.riscv_tag,):
            name = f"dma_invalid_address-{isa}-{host}-{opt}"
            tempdir = TempdirFixture()
            tests = [
                TestFunction(run_expected_invalid_address, name=name),
                verifier.MatchRegex(
                    re.compile(r".*DmaUnit: invalid source base address.*"),
                    match_stderr=True,
                    match_stdout=False,
                ).instantiate_test(name),
            ]
            TestSuite(
                name=name,
                fixtures=[build_fixture, Gem5Fixture(isa, opt, None), tempdir],
                tags=[isa, opt, constants.quick_tag, host],
                tests=tests,
            )
