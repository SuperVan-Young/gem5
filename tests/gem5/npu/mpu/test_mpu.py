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

mpu_dir = Path(__file__).resolve().parent
binary = mpu_dir / "bin" / "mpu_proto_riscv"
config_path = joinpath(
    config.base_dir,
    "tests",
    "gem5",
    "npu",
    "mpu",
    "configs",
    "mpu_proto.py",
)
build_fixture = MakeTarget(
    "all",
    make_fixture=MakeFixture(str(mpu_dir / "src")),
)


def make_named_regex_verifier(
    name, regex, *, match_stderr=True, match_stdout=True
):
    verifier_cls = type(name, (verifier.MatchRegex,), {})
    return verifier_cls(
        re.compile(regex),
        match_stderr=match_stderr,
        match_stdout=match_stdout,
    )


def add_mpu_test(name, scenario, verifier_regex):
    if isinstance(verifier_regex, str):
        verifiers = [make_named_regex_verifier(f"{name}Regex", verifier_regex)]
    else:
        verifiers = [
            make_named_regex_verifier(f"{name}Regex{idx}", regex)
            for idx, regex in enumerate(verifier_regex, start=1)
        ]

    gem5_verify_config(
        name=name,
        verifiers=verifiers,
        config=config_path,
        config_args=["--binary", str(binary), "--scenario", scenario],
        gem5_args=[
            "--debug-flags=MpuUnit,MegaCmdQueue,ScratchpadMemory,DmaUnit"
        ],
        valid_isas=(constants.riscv_tag,),
        valid_hosts=constants.supported_hosts,
        length=constants.quick_tag,
        fixtures=[build_fixture],
    )


# basic_tile_flow and output_stationary_basic have migrated to
# tests/gem5/npu/testcases/mpu/*.
add_mpu_test(
    "mpu_ab_auto_release_current_stage",
    "ab_auto_release_current_stage",
    (
        r"MPU_SUMMARY scenario=ab_auto_release_current_stage cmds=14 reads=8 writes=4 macro=0 memq=0 execq=0 drainq=0 mvin=4 load=4 compute=2 drain=2 mvout=2 a0=0 a1=0 b0=0 b1=0 c0=0 c1=0 out=0 loadedA=-1 loadedB=-1 .*",
        r"MPU_SCENARIO_PASS=ab_auto_release_current_stage",
    ),
)
add_mpu_test(
    "mpu_compute_latency_and_drain_handoff",
    "compute_latency_and_drain_handoff",
    (
        r"MPU_SUMMARY scenario=compute_latency_and_drain_handoff cmds=7 .* "
        r"compute_cycles=5 drain_cycles=[1-9][0-9]* "
        r"output_ready_cycles=13 .* macs=30 .*",
        r"MPU_SCENARIO_PASS=compute_latency_and_drain_handoff",
    ),
)
add_mpu_test(
    "mpu_drain_handoff_allows_next_compute",
    "drain_handoff_allows_next_compute",
    (
        r"MPU_SUMMARY scenario=drain_handoff_allows_next_compute cmds=14 .* "
        r"compute=2 drain=2 mvout=2 .* c0=0 c1=0 out=0 .*",
        r"MPU_SCENARIO_PASS=drain_handoff_allows_next_compute",
    ),
)
add_mpu_test(
    "mpu_spm_backpressure",
    "spm_backpressure",
    (
        r"MPU_SUMMARY scenario=spm_backpressure cmds=7 .* spm_wait=[1-9][0-9]* .*",
        r"MPU_SCENARIO_PASS=spm_backpressure",
    ),
)
add_mpu_test(
    "mpu_multi_instance_route",
    "multi_instance_route",
    (
        r"MPU_ROUTE_SUMMARY mpu0_cmds=0 mpu1_cmds=7",
        r"MPU_SCENARIO_PASS=multi_instance_route",
    ),
)
add_mpu_test(
    "mpu_dma_spm_mpu_chain",
    "dma_spm_mpu_chain",
    (
        r"MPU_DMA_SUMMARY dma_cmds=2 dma_reads=[1-9][0-9]* dma_writes=[1-9][0-9]*",
        r"MPU_SCENARIO_PASS=dma_spm_mpu_chain",
    ),
)


def run_expected_mpu_panic(params, scenario):
    fixtures = params.fixtures
    tempdir = fixtures[constants.tempdir_fixture_name].path
    gem5 = fixtures[constants.gem5_binary_fixture_name].path
    command = [
        gem5,
        "-d",
        tempdir,
        "-re",
        "--silent-redirect",
        "--debug-flags=MpuUnit,MegaCmdQueue,ScratchpadMemory,DmaUnit",
        config_path,
        "--binary",
        str(binary),
        "--scenario",
        scenario,
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
        f"Expected {scenario} scenario to terminate gem5 with a panic"
    )


def add_mpu_panic_test(name, scenario, stderr_regex):
    for host in constants.supported_hosts:
        for opt in constants.supported_variants:
            for isa in (constants.riscv_tag,):
                suite_name = f"{name}-{isa}-{host}-{opt}"
                tempdir = TempdirFixture()

                def runner(params, scenario=scenario):
                    run_expected_mpu_panic(params, scenario)

                tests = [
                    TestFunction(runner, name=suite_name),
                    verifier.MatchRegex(
                        re.compile(stderr_regex),
                        match_stderr=True,
                        match_stdout=False,
                    ).instantiate_test(suite_name),
                ]
                TestSuite(
                    name=suite_name,
                    fixtures=[
                        build_fixture,
                        Gem5Fixture(isa, opt, None),
                        tempdir,
                    ],
                    tags=[isa, opt, constants.quick_tag, host],
                    tests=tests,
                )


add_mpu_panic_test(
    "mpu_invalid_dtype",
    "invalid_dtype",
    r".*unsupported MPU data_type=1.*",
)
add_mpu_panic_test(
    "mpu_invalid_opcode",
    "invalid_opcode",
    r".*unsupported MPU command kind.*",
)
add_mpu_panic_test(
    "mpu_mvin_c_rejected",
    "mvin_c_rejected",
    r".*mvin\(C\) is not supported in MPU v1\.1.*",
)
add_mpu_panic_test(
    "mpu_load_c_rejected",
    "load_c_rejected",
    r".*load\(C\) is not supported in MPU v1\.1.*",
)
add_mpu_panic_test(
    "mpu_compute_without_loaded_inputs",
    "compute_without_loaded_inputs",
    r".*compute requires one loaded A tile and one loaded B tile.*",
)
add_mpu_panic_test(
    "mpu_drain_without_ready_output",
    "drain_without_ready_output",
    r".*drain requires output storage to be READY_TO_DRAIN.*",
)
add_mpu_panic_test(
    "mpu_mvout_without_full_c",
    "mvout_without_full_c",
    r".*mvout requires a FULL C source buffer.*",
)
