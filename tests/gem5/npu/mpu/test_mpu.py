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
        r"loops=0 matmul=0 matmul_acc=0 .* busy_mask=0 .* a0=1 .* c0=0 .*",
        r"MPU_SCENARIO_PASS=load_a",
    ),
)
add_mpu_test(
    "mpu_load_c",
    "load_c",
    (
        r"MPU_SUMMARY scenario=load_c cmds=1 loads=1 computes=0 stores=0 "
        r"loops=0 matmul=0 matmul_acc=0 .* busy_mask=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=load_c",
    ),
)
add_mpu_test(
    "mpu_matmul_basic",
    "matmul_basic",
    (
        r"MPU_SUMMARY scenario=matmul_basic cmds=4 loads=2 computes=1 "
        r"stores=1 loops=0 matmul=1 matmul_acc=0 .* "
        r"modeled_slot_stall=3000 .* observed_exec=11000 busy_mask=0 .* "
        r"c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=matmul_basic",
    ),
)
add_mpu_test(
    "mpu_matmul_acc_basic",
    "matmul_acc_basic",
    (
        r"MPU_SUMMARY scenario=matmul_acc_basic cmds=5 loads=3 computes=1 "
        r"stores=1 loops=0 matmul=0 matmul_acc=1 .* "
        r"modeled_slot_stall=5000 .* observed_exec=14000 busy_mask=0 .* "
        r"c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=matmul_acc_basic",
    ),
)
add_mpu_test(
    "mpu_sync_completion",
    "sync_completion",
    (
        r"MPU_SUMMARY scenario=sync_completion cmds=2 loads=1 computes=0 "
        r"stores=1 loops=0 matmul=0 matmul_acc=0 .* busy_mask=0 .* "
        r"c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=sync_completion",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_k_inner_local_accumulate",
    "tensor_loop_k_inner_local_accumulate",
    (
        r"MPU_SUMMARY scenario=tensor_loop_k_inner_local_accumulate cmds=1 "
        r"loads=0 computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"tiles=2 acc_tiles=1 internal_loads=4 internal_computes=2 "
        r"internal_stores=1 total_tiles=2 total_acc_tiles=1 "
        r"partial_spills=0 partial_reloads=0 modeled_spm_stall=2000 "
        r"modeled_slot_stall=5000 latency=115000 observed_load=120000 "
        r"observed_store=21000 observed_exec=25000 busy_mask=0 "
        r"tl_a_mask=1 tl_b_mask=4 tl_c_mask=16 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_k_inner_local_accumulate",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_k_outer_spill_reload",
    "tensor_loop_k_outer_spill_reload",
    (
        r"MPU_SUMMARY scenario=tensor_loop_k_outer_spill_reload cmds=1 "
        r"loads=0 computes=0 stores=0 loops=1 matmul=2 matmul_acc=2 .* "
        r"tiles=4 acc_tiles=2 internal_loads=10 internal_computes=4 "
        r"internal_stores=4 total_tiles=4 total_acc_tiles=2 "
        r"partial_spills=2 partial_reloads=2 modeled_spm_stall=8000 "
        r"modeled_slot_stall=10000 latency=284000 observed_load=486000 "
        r"observed_store=257000 observed_exec=50000 busy_mask=0 "
        r"tl_a_mask=1 tl_b_mask=4 tl_c_mask=16 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_k_outer_spill_reload",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_k_outer_pingpong_c",
    "tensor_loop_k_outer_pingpong_c",
    (
        r"MPU_SUMMARY scenario=tensor_loop_k_outer_pingpong_c cmds=1 "
        r"loads=0 computes=0 stores=0 loops=1 matmul=2 matmul_acc=2 .* "
        r"tiles=4 acc_tiles=2 internal_loads=10 internal_computes=4 "
        r"internal_stores=4 total_tiles=4 total_acc_tiles=2 "
        r"partial_spills=2 partial_reloads=2 modeled_spm_stall=8000 "
        r"modeled_slot_stall=10000 latency=284000 observed_load=486000 "
        r"observed_store=257000 observed_exec=50000 busy_mask=0 "
        r"tl_a_mask=1 tl_b_mask=4 tl_c_mask=48 .* c0=1 c1=1 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_k_outer_pingpong_c",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_matmul_acc_first_k_legacy_overwrite",
    "tensor_loop_matmul_acc_first_k_legacy_overwrite",
    (
        r"MPU_SUMMARY scenario=tensor_loop_matmul_acc_first_k_legacy_overwrite "
        r"cmds=1 loads=0 computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"tiles=2 acc_tiles=1 internal_loads=5 internal_computes=2 "
        r"internal_stores=1 total_tiles=2 total_acc_tiles=1 .* "
        r"modeled_spm_stall=4000 modeled_slot_stall=5000 latency=135000 "
        r"observed_load=180000 observed_store=21000 observed_exec=25000 "
        r"busy_mask=0 tl_a_mask=1 tl_b_mask=4 tl_c_mask=16 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_matmul_acc_first_k_legacy_overwrite",
    ),
)
add_mpu_test(
    "mpu_explicit_offset_load_store",
    "explicit_offset_load_store",
    (
        r"MPU_SUMMARY scenario=explicit_offset_load_store cmds=2 loads=1 "
        r"computes=0 stores=1 loops=0 matmul=0 matmul_acc=0 .* "
        r"busy_mask=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=explicit_offset_load_store",
    ),
)
add_mpu_test(
    "mpu_explicit_offset_compute",
    "explicit_offset_compute",
    (
        r"MPU_SUMMARY scenario=explicit_offset_compute cmds=4 loads=2 "
        r"computes=1 stores=1 loops=0 matmul=1 matmul_acc=0 .* "
        r"modeled_slot_stall=3000 .* observed_store=21000 observed_exec=11000 "
        r"busy_mask=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=explicit_offset_compute",
    ),
)
add_mpu_test(
    "mpu_tensor_loop_layout_fields_outside_step_cfg",
    "tensor_loop_layout_fields_outside_step_cfg",
    (
        r"MPU_SUMMARY scenario=tensor_loop_layout_fields_outside_step_cfg "
        r"cmds=1 loads=0 computes=0 stores=0 loops=1 matmul=1 matmul_acc=0 .* "
        r"tiles=1 acc_tiles=0 internal_loads=2 internal_computes=1 "
        r"internal_stores=1 total_tiles=1 total_acc_tiles=0 .* "
        r"modeled_spm_stall=1000 modeled_slot_stall=2000 latency=73000 "
        r"observed_load=60000 observed_store=22000 observed_exec=11000 "
        r"busy_mask=0 tl_a_mask=1 tl_b_mask=4 tl_c_mask=16 .*",
        r"MPU_SCENARIO_PASS=tensor_loop_layout_fields_outside_step_cfg",
    ),
)
add_mpu_test(
    "mpu_store_layout_conversion_normal_to_skew",
    "store_layout_conversion_normal_to_skew",
    (
        r"MPU_SUMMARY scenario=store_layout_conversion_normal_to_skew cmds=2 "
        r"loads=1 computes=0 stores=1 loops=0 matmul=0 matmul_acc=0 .* "
        r"busy_mask=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=store_layout_conversion_normal_to_skew",
    ),
)
add_mpu_test(
    "mpu_store_layout_conversion_skew_to_normal",
    "store_layout_conversion_skew_to_normal",
    (
        r"MPU_SUMMARY scenario=store_layout_conversion_skew_to_normal cmds=2 "
        r"loads=1 computes=0 stores=1 loops=0 matmul=0 matmul_acc=0 .* "
        r"busy_mask=0 .* c0=1 .* c0dirty=0 .*",
        r"MPU_SCENARIO_PASS=store_layout_conversion_skew_to_normal",
    ),
)
add_mpu_test(
    "mpu_modeled_spm_stall_and_observed_services",
    "modeled_spm_stall_and_observed_services",
    (
        r"MPU_SUMMARY scenario=modeled_spm_stall_and_observed_services cmds=1 loads=0 "
        r"computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"modeled_spm_stall=4000 modeled_slot_stall=5000 latency=135000 "
        r"observed_load=180000 observed_store=21000 observed_exec=25000 "
        r"busy_mask=0 .*",
        r"MPU_SCENARIO_PASS=modeled_spm_stall_and_observed_services",
    ),
)
add_mpu_test(
    "mpu_modeled_slot_stall_and_observed_services",
    "modeled_slot_stall_and_observed_services",
    (
        r"MPU_SUMMARY scenario=modeled_slot_stall_and_observed_services cmds=4 "
        r"loads=2 computes=1 stores=1 loops=0 matmul=1 matmul_acc=0 .* "
        r"modeled_spm_stall=0 modeled_slot_stall=3000 latency=76000 "
        r"observed_load=40000 observed_store=21000 observed_exec=11000 "
        r"busy_mask=0 .*",
        r"MPU_SCENARIO_PASS=modeled_slot_stall_and_observed_services",
    ),
)
add_mpu_test(
    "mpu_single_mem_port_tensor_loop_services",
    "single_mem_port_tensor_loop_services",
    (
        r"MPU_SUMMARY scenario=single_mem_port_tensor_loop_services cmds=1 "
        r"loads=0 computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"busy=0 active=2 tiles=2 acc_tiles=1 internal_loads=5 "
        r"internal_computes=2 "
        r"internal_stores=1 .* modeled_spm_stall=4000 "
        r"modeled_slot_stall=5000 latency=135000 observed_load=180000 "
        r"observed_store=21000 observed_exec=25000 busy_mask=0 .*",
        r"MPU_SCENARIO_PASS=single_mem_port_tensor_loop_services",
    ),
)
add_mpu_test(
    "mpu_multi_mem_port_tensor_loop_services",
    "multi_mem_port_tensor_loop_services",
    (
        r"MPU_SUMMARY scenario=multi_mem_port_tensor_loop_services cmds=1 "
        r"loads=0 computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"busy=0 active=3 tiles=2 acc_tiles=1 internal_loads=5 "
        r"internal_computes=2 "
        r"internal_stores=1 .* modeled_spm_stall=0 "
        r"modeled_slot_stall=5000 latency=78000 observed_load=104000 "
        r"observed_store=21000 observed_exec=25000 busy_mask=0 .*",
        r"MPU_SCENARIO_PASS=multi_mem_port_tensor_loop_services",
    ),
)
add_mpu_test(
    "mpu_geometry_default_compute_latency",
    "geometry_default_compute_latency",
    (
        r"MPU_SUMMARY scenario=geometry_default_compute_latency cmds=1 "
        r"loads=0 computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"modeled_spm_stall=2000 modeled_slot_stall=5000 latency=115000 "
        r"observed_load=120000 observed_store=21000 observed_exec=25000 "
        r"busy_mask=0 .*",
        r"MPU_SCENARIO_PASS=geometry_default_compute_latency",
    ),
)
add_mpu_test(
    "mpu_geometry_override_compute_latency",
    "geometry_override_compute_latency",
    (
        r"MPU_SUMMARY scenario=geometry_override_compute_latency cmds=1 "
        r"loads=0 computes=0 stores=0 loops=1 matmul=1 matmul_acc=1 .* "
        r"modeled_spm_stall=2000 modeled_slot_stall=5000 latency=141000 "
        r"observed_load=120000 observed_store=21000 observed_exec=77000 "
        r"busy_mask=0 .*",
        r"MPU_SCENARIO_PASS=geometry_override_compute_latency",
    ),
)
add_mpu_test(
    "mpu_dma_to_mpu_to_dma_regression",
    "dma_to_mpu_to_dma_regression",
    (
        r"MPU_SUMMARY scenario=dma_to_mpu_to_dma_regression cmds=4 loads=2 "
        r"computes=1 stores=1 loops=0 matmul=1 matmul_acc=0 .*",
        r"DMA_SUMMARY scenario=dma_to_mpu_to_dma_regression cmds=3 .*",
        r"MPU_SCENARIO_PASS=dma_to_mpu_to_dma_regression",
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
    "mpu_low_bits_in_local_addr_rejected",
    "low_bits_in_local_addr_rejected",
    r".*MpuUnit: load destination local address .* must use slot base "
    r"without low-bit offset.*",
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
