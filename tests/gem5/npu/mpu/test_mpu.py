# Copyright (c) 2026
# All rights reserved.

import re
import subprocess
from pathlib import Path

from testlib import *

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
        re.compile(regex, re.DOTALL),
        match_stderr=match_stderr,
        match_stdout=match_stdout,
    )


def add_mpu_test(name, scenario, verifier_regex):
    if isinstance(verifier_regex, str):
        verifiers = [make_named_regex_verifier(f"{name}Regex", verifier_regex)]
    else:
        verifiers = [
            make_named_regex_verifier(f"{name}Regex{index}", regex)
            for index, regex in enumerate(verifier_regex, start=1)
        ]

    gem5_verify_config(
        name=name,
        verifiers=verifiers,
        config=config_path,
        config_args=["--binary", str(binary), "--scenario", scenario],
        gem5_args=["--debug-flags=MegaCmdQueue,MpuUnit,DmaUnit"],
        valid_isas=(constants.riscv_tag,),
        valid_hosts=constants.supported_hosts,
        length=constants.quick_tag,
        fixtures=[build_fixture],
    )


add_mpu_test(
    "mpu_load_a",
    "load_a",
    (
        r"MPU_SUMMARY scenario=load_a cmds=1 loads=1 computes=0 stores=0 "
        r"loops=0 matmul=0 matmul_acc=0 .* a0=1 .* c0=0 .*",
        r"MPU_SCENARIO_PASS=load_a",
    ),
)
add_mpu_test(
    "mpu_load_c",
    "load_c",
    (
        r"MPU_SUMMARY scenario=load_c cmds=1 loads=1 computes=0 stores=0 "
        r"loops=0 matmul=0 matmul_acc=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=load_c",
    ),
)
add_mpu_test(
    "mpu_matmul_basic",
    "matmul_basic",
    (
        r"MPU_SUMMARY scenario=matmul_basic cmds=4 loads=2 computes=1 "
        r"stores=1 loops=0 matmul=1 matmul_acc=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=matmul_basic",
    ),
)
add_mpu_test(
    "mpu_matmul_acc_basic",
    "matmul_acc_basic",
    (
        r"MPU_SUMMARY scenario=matmul_acc_basic cmds=5 loads=3 computes=1 "
        r"stores=1 loops=0 matmul=0 matmul_acc=1 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=matmul_acc_basic",
    ),
)
add_mpu_test(
    "mpu_store_basic",
    "store_basic",
    (
        r"MPU_SUMMARY scenario=store_basic cmds=2 loads=1 computes=0 "
        r"stores=1 loops=0 matmul=0 matmul_acc=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=store_basic",
    ),
)
add_mpu_test(
    "mpu_sync_completion",
    "sync_completion",
    (
        r"MPU_SUMMARY scenario=sync_completion cmds=2 loads=1 computes=0 "
        r"stores=1 loops=0 matmul=0 matmul_acc=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=sync_completion",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_mn_basic",
    "tensor_loop_mn_basic",
    (
        r"MPU_SUMMARY scenario=tensor_loop_mn_basic cmds=1 loads=0 "
        r"computes=0 stores=0 loops=1 matmul=4 matmul_acc=0 .* "
        r"tiles=4 acc_tiles=0 internal_loads=8 internal_computes=4 "
        r"internal_stores=4 total_tiles=4 total_acc_tiles=0 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_mn_basic",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_k_matmul_basic",
    "tensor_loop_k_matmul_basic",
    (
        r"MPU_SUMMARY scenario=tensor_loop_k_matmul_basic cmds=1 loads=0 "
        r"computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"tiles=2 acc_tiles=1 internal_loads=4 internal_computes=2 "
        r"internal_stores=1 total_tiles=2 total_acc_tiles=1 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_k_matmul_basic",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_k_matmul_acc_header_overwrite_old_c",
    "tensor_loop_k_matmul_acc_header_overwrite_old_c",
    (
        r"MPU_SUMMARY scenario=tensor_loop_k_matmul_acc_header_overwrite_old_c.*"
        r"matmul=1 matmul_acc=1.*internal_loads=5.*internal_computes=2.*"
        r"internal_stores=1.*total_acc_tiles=1.*spm_stall=[1-9][0-9]*.*",
        r"MPU_SCENARIO_PASS=tensor_loop_k_matmul_acc_header_overwrite_old_c",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_pingpong_ab",
    "tensor_loop_pingpong_ab",
    (
        r"MPU_SUMMARY scenario=tensor_loop_pingpong_ab cmds=1 loads=0 "
        r"computes=0 stores=0 loops=1 matmul=4 matmul_acc=0 .* "
        r"tiles=4 acc_tiles=0 internal_loads=8 internal_computes=4 "
        r"internal_stores=4 .* a0=1 a1=1 b0=1 b1=1 c0=1 c1=0 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_pingpong_ab",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_pingpong_c_per_output_tile",
    "tensor_loop_pingpong_c_per_output_tile",
    (
        r"MPU_SUMMARY scenario=tensor_loop_pingpong_c_per_output_tile cmds=1 "
        r"loads=0 computes=0 stores=0 loops=1 matmul=2 matmul_acc=0 .* "
        r"tiles=2 acc_tiles=0 internal_loads=4 internal_computes=2 "
        r"internal_stores=2 .* c0=1 c1=1 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_pingpong_c_per_output_tile",
    ),
)
add_mpu_test(
    "mpu_nonzero_local_offset_load_store",
    "nonzero_local_offset_load_store",
    (
        r"MPU_SUMMARY scenario=nonzero_local_offset_load_store cmds=2 "
        r"loads=1 computes=0 stores=1 loops=0 matmul=0 matmul_acc=0 .* "
        r"c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=nonzero_local_offset_load_store",
    ),
)
add_mpu_test(
    "mpu_nonzero_local_offset_compute",
    "nonzero_local_offset_compute",
    (
        r"MPU_SUMMARY scenario=nonzero_local_offset_compute cmds=4 loads=2 "
        r"computes=1 stores=1 loops=0 matmul=1 matmul_acc=0 .* "
        r"slot_stall=[1-9][0-9]* .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=nonzero_local_offset_compute",
    ),
)
add_mpu_test(
    "mpu_skewed_layout_roundtrip",
    "skewed_layout_roundtrip",
    (
        r"MPU_SUMMARY scenario=skewed_layout_roundtrip cmds=2 loads=1 "
        r"computes=0 stores=1 loops=0 matmul=0 matmul_acc=0 .* "
        r"c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=skewed_layout_roundtrip",
    ),
)
add_mpu_test(
    "mpu_local_bank_conflict_stall_stats",
    "local_bank_conflict_stall_stats",
    (
        r"MPU_SUMMARY scenario=local_bank_conflict_stall_stats cmds=4 loads=2 "
        r"computes=1 stores=1 loops=0 matmul=1 matmul_acc=0 .* "
        r"slot_stall=[1-9][0-9]* latency=[1-9][0-9]* .*",
        r"MPU_SCENARIO_PASS=local_bank_conflict_stall_stats",
    ),
)
add_mpu_test(
    "mpu_spm_backpressure_stall_stats",
    "spm_backpressure_stall_stats",
    (
        r"MPU_SUMMARY scenario=spm_backpressure_stall_stats cmds=1 loads=0 "
        r"computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"spm_stall=[1-9][0-9]* slot_stall=[1-9][0-9]* latency=[1-9][0-9]* .*",
        r"MPU_SCENARIO_PASS=spm_backpressure_stall_stats",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_stats_latency",
    "tensor_loop_stats_latency",
    (
        r"MPU_SUMMARY scenario=tensor_loop_stats_latency cmds=1 loads=0 "
        r"computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"tiles=2 acc_tiles=1 .* latency=[1-9][0-9]* .*",
        r"MPU_SCENARIO_PASS=tensor_loop_stats_latency",
    ),
)
add_mpu_test(
    "mpu_dma_chain",
    "dma_chain",
    (
        r"MPU_SUMMARY scenario=dma_chain cmds=4 loads=2 computes=1 stores=1 "
        r"loops=0 matmul=1 matmul_acc=0 .*",
        r"DMA_SUMMARY scenario=dma_chain cmds=3 reads=6 writes=3 iters=3 .*",
        r"MPU_SCENARIO_PASS=dma_chain",
    ),
)


def run_expected_mpu_panic(params, scenario, stderr_regex):
    fixtures = params.fixtures
    tempdir = fixtures[constants.tempdir_fixture_name].path
    gem5 = fixtures[constants.gem5_binary_fixture_name].path
    command = [
        gem5,
        "-d",
        tempdir,
        "-re",
        "--silent-redirect",
        "--debug-flags=MegaCmdQueue,MpuUnit,DmaUnit",
        config_path,
        "--binary",
        str(binary),
        "--scenario",
        scenario,
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    simout = Path(tempdir) / "simout.txt"
    simerr = Path(tempdir) / "simerr.txt"
    stderr = simerr.read_text() if simerr.exists() else completed.stderr
    stdout = simout.read_text() if simout.exists() else completed.stdout

    if completed.returncode == 0:
        raise AssertionError(
            f"Expected {scenario} scenario to terminate gem5 with a panic"
        )
    if not re.search(stderr_regex, stderr, re.DOTALL):
        raise AssertionError(
            f"Expected panic regex {stderr_regex!r} for {scenario}, got stderr\n"
            f"{stderr}\nSTDOUT:\n{stdout}"
        )


def add_mpu_panic_test(name, scenario, stderr_regex):
    for host in constants.supported_hosts:
        for opt in constants.supported_variants:
            for isa in (constants.riscv_tag,):
                suite_name = f"{name}-{isa}-{host}-{opt}"
                tempdir = TempdirFixture()

                def runner(
                    params, scenario=scenario, stderr_regex=stderr_regex
                ):
                    run_expected_mpu_panic(params, scenario, stderr_regex)

                TestSuite(
                    name=suite_name,
                    fixtures=[
                        build_fixture,
                        Gem5Fixture(isa, opt, None),
                        tempdir,
                    ],
                    tags=[isa, opt, constants.quick_tag, host],
                    tests=[TestFunction(runner, name=suite_name)],
                )


add_mpu_panic_test(
    "mpu_invalid_local_addr",
    "invalid_local_addr",
    r".*MpuUnit: invalid load destination local address.*",
)
add_mpu_panic_test(
    "mpu_matmul_acc_missing_c",
    "matmul_acc_missing_c",
    r".*MpuUnit: MATMUL_ACC requires valid C slot.*",
)
add_mpu_panic_test(
    "mpu_tensor_loop_auto_load_c_for_acc_required",
    "tensor_loop_auto_load_c_for_acc_required",
    r".*MpuUnit: tensor_loop MATMUL_ACC requires auto_load_c_for_acc=1.*",
)
