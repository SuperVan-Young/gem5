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


def add_dma_test(name, scenario, verifier_regex):
    gem5_verify_config(
        name=name,
        verifiers=[verifier.MatchRegex(re.compile(verifier_regex))],
        config=config_path,
        config_args=["--binary", str(binary), "--scenario", scenario],
        gem5_args=["--debug-flags=DmaUnit,MegaCmdQueue"],
        valid_isas=(constants.riscv_tag,),
        valid_hosts=constants.supported_hosts,
        length=constants.quick_tag,
        fixtures=[build_fixture],
    )


add_dma_test(
    "dma_basic_dram_to_spm",
    "basic_dram_to_spm",
    r"DMA_SCENARIO_PASS=basic_dram_to_spm",
)
add_dma_test(
    "dma_basic_spm_to_dram",
    "basic_spm_to_dram",
    r"DMA_SCENARIO_PASS=basic_spm_to_dram",
)
add_dma_test(
    "dma_hwc_to_blocked", "hwc_to_blocked", r"DMA_SCENARIO_PASS=hwc_to_blocked"
)
add_dma_test(
    "dma_blocked_to_blocked",
    "blocked_to_blocked",
    r"DMA_SCENARIO_PASS=blocked_to_blocked",
)
add_dma_test(
    "dma_buffer_size_forces_batching",
    "buffer_size_forces_batching",
    r"DMA_SCENARIO_PASS=buffer_size_forces_batching",
)
add_dma_test(
    "dma_sync_completion",
    "sync_completion",
    r"DMA_SCENARIO_PASS=sync_completion",
)
add_dma_test(
    "dma_queued_chain",
    "queued_chain",
    (
        r"(?s)DMA_SUMMARY scenario=queued_chain cmds=2 .* "
        r"queue=0 cmdq=0 busy=0.*DMA_SCENARIO_PASS=queued_chain"
    ),
)


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
        "--debug-flags=DmaUnit,MegaCmdQueue",
        config_path,
        "--binary",
        str(binary),
        "--scenario",
        "invalid_address",
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
