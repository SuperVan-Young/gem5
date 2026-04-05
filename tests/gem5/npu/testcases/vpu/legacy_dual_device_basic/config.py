# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import (  # noqa: E402
    collect_component_snapshot,
    emit_summary,
    verify_snapshot,
)
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_VPU0 = {
    "completed_cmds": 2,
    "prologues": 3,
    "executes": 3,
    "epilogues": 3,
    "read_resps": 5,
    "write_resps": 3,
    "iterations": 3,
}
EXPECTED_VPU1 = dict(EXPECTED_VPU0)

VPU_FIELDS = {
    "completed_cmds": "completedCmdCount",
    "prologues": "prologueCount",
    "executes": "executeCount",
    "epilogues": "epilogueCount",
    "read_resps": "completedReadRespCount",
    "write_resps": "completedWriteRespCount",
    "iterations": "completedIterationCount",
}

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()


def build_m5_system(args):
    binary = os.path.abspath(args.binary)
    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_spm()
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(binary, cpu_id=0)
    builder.add_megacmdqueue()
    builder.add_vpu(vpu_id=0, num_mem_side_ports=4)
    builder.add_vpu(vpu_id=1, num_mem_side_ports=4)
    return builder, process


def collect_vpu_snapshot(component, prefix):
    snapshot = collect_component_snapshot(component, VPU_FIELDS)
    return {f"{prefix}_{key}": value for key, value in snapshot.items()}


def collect_simulation_result(builder, exit_cause):
    snapshot = {"exit_cause": exit_cause}
    snapshot.update(collect_vpu_snapshot(builder.system.vpu0, "vpu0"))
    snapshot.update(collect_vpu_snapshot(builder.system.vpu1, "vpu1"))
    return snapshot


def verify_simulation_result(snapshot):
    return verify_snapshot(
        snapshot,
        {
            "exit_cause": EXPECTED_EXIT_CAUSE,
            "vpu0_completed_cmds": EXPECTED_VPU0["completed_cmds"],
            "vpu0_prologues": EXPECTED_VPU0["prologues"],
            "vpu0_executes": EXPECTED_VPU0["executes"],
            "vpu0_epilogues": EXPECTED_VPU0["epilogues"],
            "vpu0_read_resps": EXPECTED_VPU0["read_resps"],
            "vpu0_write_resps": EXPECTED_VPU0["write_resps"],
            "vpu0_iterations": EXPECTED_VPU0["iterations"],
            "vpu1_completed_cmds": EXPECTED_VPU1["completed_cmds"],
            "vpu1_prologues": EXPECTED_VPU1["prologues"],
            "vpu1_executes": EXPECTED_VPU1["executes"],
            "vpu1_epilogues": EXPECTED_VPU1["epilogues"],
            "vpu1_read_resps": EXPECTED_VPU1["read_resps"],
            "vpu1_write_resps": EXPECTED_VPU1["write_resps"],
            "vpu1_iterations": EXPECTED_VPU1["iterations"],
        },
    )


builder, process = build_m5_system(args)
root = builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_spm(process=process)
builder.map_vpu(vpu_id=0, process=process)
builder.map_vpu(vpu_id=1, process=process)

exit_event = m5.simulate()
result = collect_simulation_result(builder, exit_event.getCause())
emit_summary("VPU_LEGACY_DUAL_DEVICE_BASIC", result)

if verify_simulation_result(result):
    print("VPU_LEGACY_DUAL_DEVICE_BASIC_PASS")
