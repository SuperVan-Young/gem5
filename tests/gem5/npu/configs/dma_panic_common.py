# Copyright (c) 2026
# All rights reserved.

"""Shared harness helpers for DMA reject testcases.

Each reject testcase keeps its local files minimal:

- ``test.py`` only registers the expected-panic suite,
- ``config.py`` only forwards to ``run_dma_panic_config(...)``,
- ``workload.c`` only selects the single illegal DMA command shape.

The shared layer centralizes the scenario-to-regex mapping and the
expected-panic runner so each testcase still covers one rejection rule only.
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DmaPanicCaseSpec:
    scenario: str
    stderr_regex: str


DMA_PANIC_CASES = {
    "reject_invalid_source_address": DmaPanicCaseSpec(
        scenario="invalid_source_address",
        stderr_regex=r".*DmaUnit: invalid source base address.*",
    ),
    "reject_invalid_destination_address": DmaPanicCaseSpec(
        scenario="invalid_destination_address",
        stderr_regex=r".*DmaUnit: invalid destination base address.*",
    ),
    "reject_invalid_blocked_k": DmaPanicCaseSpec(
        scenario="invalid_blocked_k",
        stderr_regex=r".*DmaUnit: source blocked layout requires W % k == 0.*",
    ),
    "reject_invalid_blocked_k_h": DmaPanicCaseSpec(
        scenario="invalid_blocked_k_h",
        stderr_regex=r".*DmaUnit: source blocked layout requires H % k == 0.*",
    ),
    "reject_invalid_blocked_k_c": DmaPanicCaseSpec(
        scenario="invalid_blocked_k_c",
        stderr_regex=r".*DmaUnit: source blocked layout requires C % k == 0.*",
    ),
    "reject_unsupported_data_type": DmaPanicCaseSpec(
        scenario="unsupported_data_type",
        stderr_regex=r".*DmaUnit: unsupported data_type=1.*",
    ),
    "reject_reserved_mode": DmaPanicCaseSpec(
        scenario="reserved_mode",
        stderr_regex=r".*DmaUnit: unsupported mode=3.*",
    ),
    "reject_reserved_cut_dim": DmaPanicCaseSpec(
        scenario="reserved_cut_dim",
        stderr_regex=r".*DmaUnit: reserved src_cut_dim=3.*",
    ),
    "reject_reserved_transpose_dim": DmaPanicCaseSpec(
        scenario="reserved_transpose_dim",
        stderr_regex=r".*DmaUnit: reserved transpose_dim_a=3.*",
    ),
    "reject_reserved_bank_cfg_bits": DmaPanicCaseSpec(
        scenario="reserved_bank_cfg_bits",
        stderr_regex=r".*DmaUnit: reserved bank_cfg bits set for transpose.*",
    ),
    "reject_out_of_range_bank_id": DmaPanicCaseSpec(
        scenario="out_of_range_bank_id",
        stderr_regex=r".*DmaUnit: src_bank_id=2 exceeds num_banks=2.*",
    ),
    "reject_memory_space_mismatch": DmaPanicCaseSpec(
        scenario="memory_space_mismatch",
        stderr_regex=r".*DmaUnit: invalid source base address 0x60001000.*",
    ),
    "reject_fill_invalid_bank_id": DmaPanicCaseSpec(
        scenario="fill_invalid_bank_id",
        stderr_regex=r".*DmaUnit: dst_bank_id=2 exceeds num_banks=2.*",
    ),
    "reject_fill_reserved_bank_cfg_bits": DmaPanicCaseSpec(
        scenario="fill_reserved_bank_cfg_bits",
        stderr_regex=r".*DmaUnit: reserved bank_cfg bits set for fill.*",
    ),
    "reject_fill_invalid_contract": DmaPanicCaseSpec(
        scenario="fill_invalid_contract",
        stderr_regex=r".*DmaUnit: fill requires src_base_addr == 0.*",
    ),
    "reject_fill_exceeds_bank_size": DmaPanicCaseSpec(
        scenario="fill_exceeds_bank_size",
        stderr_regex=(
            r".*DmaUnit: fill required_bytes=4097 exceeds "
            r"bank_size=4096.*"
        ),
    ),
    "reject_fill_invalid_dst_mem_space": DmaPanicCaseSpec(
        scenario="fill_invalid_dst_mem_space",
        stderr_regex=r".*DmaUnit: reserved dst_mem_space=3 for fill.*",
    ),
    "reject_fill_partial_line_dram": DmaPanicCaseSpec(
        scenario="fill_partial_line_dram",
        stderr_regex=(
            r".*DmaUnit: external fill requires full 64B "
            r"cache-line coverage.*"
        ),
    ),
    "reject_fill_partial_line_spm": DmaPanicCaseSpec(
        scenario="fill_partial_line_spm",
        stderr_regex=(
            r".*DmaUnit: external fill requires full 64B "
            r"cache-line coverage.*"
        ),
    ),
    "reject_fill_reserved_fill_value_bits": DmaPanicCaseSpec(
        scenario="fill_reserved_fill_value_bits",
        stderr_regex=r".*DmaUnit: fill requires Word 15\[31:8\] == 0.*",
    ),
    "reject_transpose_same_bank": DmaPanicCaseSpec(
        scenario="transpose_same_bank",
        stderr_regex=(
            r".*DmaUnit: transpose requires src_bank_id != "
            r"dst_bank_id.*"
        ),
    ),
    "reject_transpose_equal_dims": DmaPanicCaseSpec(
        scenario="transpose_equal_dims",
        stderr_regex=(
            r".*DmaUnit: transpose requires transpose_dim_a != "
            r"transpose_dim_b.*"
        ),
    ),
    "reject_transpose_nonzero_k": DmaPanicCaseSpec(
        scenario="transpose_nonzero_k",
        stderr_regex=(
            r".*DmaUnit: transpose requires src_k == 0 and "
            r"dst_k == 0.*"
        ),
    ),
    "reject_transpose_exceeds_bank_size": DmaPanicCaseSpec(
        scenario="transpose_exceeds_bank_size",
        stderr_regex=(
            r".*DmaUnit: transpose required_bytes=4097 exceeds "
            r"bank_size=4096.*"
        ),
    ),
}


def get_dma_panic_case(case_name):
    try:
        return DMA_PANIC_CASES[case_name]
    except KeyError as exc:
        raise KeyError(f"Unknown DMA panic testcase '{case_name}'") from exc


def _run_expected_panic(params, config_path, binary_path, scenario):
    from testlib.configuration import constants
    from testlib.helper import log_call

    tempdir = params.fixtures[constants.tempdir_fixture_name].path
    gem5 = params.fixtures[constants.gem5_binary_fixture_name].path
    command = [
        gem5,
        "-d",
        tempdir,
        "-re",
        "--silent-redirect",
        "--debug-flags=DmaUnit,MegaCmdQueue",
        config_path,
        "--binary",
        binary_path,
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


def register_dma_panic_test(reference_file, case_name):
    from gem5 import verifier
    from gem5.fixture import Gem5Fixture, TempdirFixture
    from testlib import TestFunction, TestSuite
    from testlib.configuration import constants

    from runner_common import (
        make_testcase_build_fixture,
        resolve_binary_path,
        resolve_config_path,
    )

    spec = get_dma_panic_case(case_name)
    config_path = str(resolve_config_path(reference_file))
    binary_path = str(
        resolve_binary_path(reference_file, f"dma_{case_name}_riscv")
    )
    build_fixture = make_testcase_build_fixture(reference_file)

    for host in constants.supported_hosts:
        for opt in constants.supported_variants:
            for isa in (constants.riscv_tag,):
                suite_name = f"{case_name}-{isa}-{host}-{opt}"
                tempdir = TempdirFixture()

                def runner(params, scenario=spec.scenario):
                    # The gem5 process must abort before control returns here.
                    _run_expected_panic(
                        params,
                        config_path,
                        binary_path,
                        scenario,
                    )

                tests = [
                    TestFunction(runner, name=suite_name),
                    verifier.MatchRegex(
                        re.compile(spec.stderr_regex),
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


def build_dma_panic_system(binary, scenario):
    from npu_test_system import NPUTestSystemBuilder

    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_spm()
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(os.path.abspath(binary), argv=[scenario])
    builder.add_megacmdqueue()
    builder.add_dma()
    return builder, process


def run_dma_panic_config(case_name):
    import m5

    spec = get_dma_panic_case(case_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()

    binary = os.path.abspath(args.binary)
    builder, process = build_dma_panic_system(binary, spec.scenario)
    builder.instantiate_root()
    m5.instantiate()
    builder.map_cmdq(process=process)
    builder.map_spm(process=process)
    builder.map_dram(process=process)

    m5.simulate()
    raise AssertionError(
        f"Expected {case_name} scenario to terminate gem5 with a panic"
    )
