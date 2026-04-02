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
        r"loops=0 matmul=0 matmul_acc=0 reads=1 writes=0 iters=1 .* "
        r"a0=1 .* c0=0 .*",
        r"MPU_SCENARIO_PASS=load_a",
    ),
)
add_mpu_test(
    "mpu_load_c",
    "load_c",
    (
        r"MPU_SUMMARY scenario=load_c cmds=1 loads=1 computes=0 stores=0 "
        r"loops=0 matmul=0 matmul_acc=0 reads=1 writes=0 iters=1 .* "
        r"c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=load_c",
    ),
)
add_mpu_test(
    "mpu_matmul_basic",
    "matmul_basic",
    (
        r"MPU_SUMMARY scenario=matmul_basic cmds=4 loads=2 computes=1 "
        r"stores=1 loops=0 matmul=1 matmul_acc=0 reads=2 writes=1 "
        r"iters=4 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=matmul_basic",
    ),
)
add_mpu_test(
    "mpu_matmul_acc_basic",
    "matmul_acc_basic",
    (
        r"MPU_SUMMARY scenario=matmul_acc_basic cmds=5 loads=3 computes=1 "
        r"stores=1 loops=0 matmul=0 matmul_acc=1 reads=3 writes=1 "
        r"iters=5 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=matmul_acc_basic",
    ),
)
add_mpu_test(
    "mpu_store_basic",
    "store_basic",
    (
        r"MPU_SUMMARY scenario=store_basic cmds=2 loads=1 computes=0 "
        r"stores=1 loops=0 matmul=0 matmul_acc=0 reads=1 writes=1 "
        r"iters=2 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=store_basic",
    ),
)
add_mpu_test(
    "mpu_sync_completion",
    "sync_completion",
    (
        r"MPU_SUMMARY scenario=sync_completion cmds=2 loads=1 computes=0 "
        r"stores=1 loops=0 matmul=0 matmul_acc=0 reads=1 writes=1 "
        r"iters=2 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=sync_completion",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_mn_basic",
    "tensor_loop_mn_basic",
    (
        r"MPU_SUMMARY scenario=tensor_loop_mn_basic cmds=1 loads=0 "
        r"computes=0 stores=0 loops=1 matmul=4 matmul_acc=0 reads=8 "
        r"writes=4 iters=4 .* tiles=4 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_mn_basic",
    ),
)
add_mpu_test(
    "mpu_dma_chain",
    "dma_chain",
    (
        r"MPU_SUMMARY scenario=dma_chain cmds=4 loads=2 computes=1 stores=1 "
        r"loops=0 matmul=1 matmul_acc=0 reads=2 writes=1 iters=4 .*",
        r"DMA_SUMMARY scenario=dma_chain cmds=3 reads=6 writes=3 iters=3 .*",
        r"MPU_SCENARIO_PASS=dma_chain",
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
        "--debug-flags=MegaCmdQueue,MpuUnit,DmaUnit",
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
    "mpu_tensor_loop_k_axis",
    "tensor_loop_k_axis",
    r".*MpuUnit: Phase 3 tensor_loop does not support K-axis expansion.*",
)
