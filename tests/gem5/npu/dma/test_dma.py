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
add_dma_test("dma_spm_to_spm", "spm_to_spm", r"DMA_SCENARIO_PASS=spm_to_spm")
add_dma_test(
    "dma_dram_to_dram", "dram_to_dram", r"DMA_SCENARIO_PASS=dram_to_dram"
)
add_dma_test("dma_cut_dim_h", "cut_dim_h", r"DMA_SCENARIO_PASS=cut_dim_h")
add_dma_test("dma_cut_dim_w", "cut_dim_w", r"DMA_SCENARIO_PASS=cut_dim_w")
add_dma_test("dma_cut_dim_c", "cut_dim_c", r"DMA_SCENARIO_PASS=cut_dim_c")
add_dma_test(
    "dma_fill_zero_bank",
    "fill_zero_bank",
    (
        r"(?s)DMA_SUMMARY scenario=fill_zero_bank cmds=1 reads=0 writes=0 .*"
        r"DMA_SCENARIO_PASS=fill_zero_bank"
    ),
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


def run_expected_dma_panic(params, scenario):
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


def add_dma_panic_test(name, scenario, stderr_regex):
    for host in constants.supported_hosts:
        for opt in constants.supported_variants:
            for isa in (constants.riscv_tag,):
                suite_name = f"{name}-{isa}-{host}-{opt}"
                tempdir = TempdirFixture()

                def runner(params, scenario=scenario):
                    run_expected_dma_panic(params, scenario)

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


add_dma_panic_test(
    "dma_invalid_address",
    "invalid_address",
    r".*DmaUnit: invalid source base address.*",
)
add_dma_panic_test(
    "dma_invalid_destination_address",
    "invalid_destination_address",
    r".*DmaUnit: invalid destination base address.*",
)
add_dma_panic_test(
    "dma_invalid_blocked_k",
    "invalid_blocked_k",
    r".*DmaUnit: source blocked layout requires W % k == 0.*",
)
add_dma_panic_test(
    "dma_invalid_blocked_k_h",
    "invalid_blocked_k_h",
    r".*DmaUnit: source blocked layout requires H % k == 0.*",
)
add_dma_panic_test(
    "dma_invalid_blocked_k_c",
    "invalid_blocked_k_c",
    r".*DmaUnit: source blocked layout requires C % k == 0.*",
)
add_dma_panic_test(
    "dma_unsupported_data_type",
    "unsupported_data_type",
    r".*DmaUnit: unsupported data_type=1.*",
)
add_dma_panic_test(
    "dma_reserved_mode",
    "reserved_mode",
    r".*DmaUnit: unsupported mode=3.*",
)
add_dma_panic_test(
    "dma_reserved_cut_dim",
    "reserved_cut_dim",
    r".*DmaUnit: reserved src_cut_dim=3.*",
)
add_dma_panic_test(
    "dma_reserved_transpose_dim",
    "reserved_transpose_dim",
    r".*DmaUnit: reserved transpose_dim_a=3.*",
)
add_dma_panic_test(
    "dma_reserved_bank_cfg_bits",
    "reserved_bank_cfg_bits",
    r".*DmaUnit: reserved bank_cfg bits set for transpose.*",
)
add_dma_panic_test(
    "dma_out_of_range_bank_id",
    "out_of_range_bank_id",
    r".*DmaUnit: src_bank_id=2 exceeds num_banks=2.*",
)
add_dma_panic_test(
    "dma_memory_space_mismatch",
    "memory_space_mismatch",
    r".*DmaUnit: invalid source base address 0x60001000.*",
)
add_dma_panic_test(
    "dma_fill_invalid_bank_id",
    "fill_invalid_bank_id",
    r".*DmaUnit: dst_bank_id=2 exceeds num_banks=2.*",
)
add_dma_panic_test(
    "dma_fill_reserved_bank_cfg_bits",
    "fill_reserved_bank_cfg_bits",
    r".*DmaUnit: reserved bank_cfg bits set for fill.*",
)
add_dma_panic_test(
    "dma_fill_invalid_contract",
    "fill_invalid_contract",
    r".*DmaUnit: fill requires src_base_addr == 0.*",
)
add_dma_panic_test(
    "dma_fill_exceeds_bank_size",
    "fill_exceeds_bank_size",
    r".*DmaUnit: fill required_bytes=4097 exceeds bank_size=4096.*",
)
