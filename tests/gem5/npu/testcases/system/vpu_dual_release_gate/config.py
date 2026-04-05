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
EXPECTED_EXIT_CODE = 0
EXPECTED_SHARED_LUT_REQUESTS = 4
EXPECTED_SHARED_LUT_COMMANDS = 1
EXPECTED_VPU0 = {
    "completed_cmds": 1,
    "prologues": 1,
    "executes": 1,
    "epilogues": 1,
    "iterations": 1,
    "read_resps": 1,
    "write_resps": 1,
}
EXPECTED_VPU1 = dict(EXPECTED_VPU0)

VPU_FIELDS = {
    "completed_cmds": "completedCmdCount",
    "prologues": "prologueCount",
    "executes": "executeCount",
    "epilogues": "epilogueCount",
    "iterations": "completedIterationCount",
    "read_resps": "completedReadRespCount",
    "write_resps": "completedWriteRespCount",
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
    seu = builder.add_mega_seu(
        num_vpus=2,
        include_dma=False,
        vpu_num_mem_side_ports=4,
    )
    return builder, process, seu


def collect_vpu_snapshot(component, prefix):
    snapshot = collect_component_snapshot(component, VPU_FIELDS)
    return {f"{prefix}_{key}": value for key, value in snapshot.items()}


def collect_simulation_result(builder, seu, exit_event):
    snapshot = {
        "exit_cause": exit_event.getCause(),
        "exit_code": exit_event.getCode(),
        "cmdq_occupancy": builder.system.cmdq.queueOccupancy(),
        "vpu0_occupancy": seu.vpu0.queueOccupancy(),
        "vpu1_occupancy": seu.vpu1.queueOccupancy(),
        "vpu0_busy": seu.vpu0.isIssueBusy(),
        "vpu1_busy": seu.vpu1.isIssueBusy(),
        "shared_lut_requests": seu.lut.requestCount(),
        "shared_lut_commands": seu.lut.commandCount(),
        "shared_lut_latency": seu.lut.lastExecuteLatency(),
        "shared_lut_completion_tick": seu.lut.lastCompletionTick(),
        "vpu0_linear_completion_tick": seu.vpu0.lastLinearCompletionTick(),
        "vpu1_lut_completion_tick": seu.vpu1.lastLutCompletionTick(),
    }
    snapshot.update(collect_vpu_snapshot(seu.vpu0, "vpu0"))
    snapshot.update(collect_vpu_snapshot(seu.vpu1, "vpu1"))
    return snapshot


def verify_simulation_result(snapshot):
    verified = verify_snapshot(
        snapshot,
        {
            "exit_cause": EXPECTED_EXIT_CAUSE,
            "exit_code": EXPECTED_EXIT_CODE,
            "cmdq_occupancy": 0,
            "vpu0_occupancy": 0,
            "vpu1_occupancy": 0,
            "vpu0_busy": False,
            "vpu1_busy": False,
            "vpu0_completed_cmds": EXPECTED_VPU0["completed_cmds"],
            "vpu0_prologues": EXPECTED_VPU0["prologues"],
            "vpu0_executes": EXPECTED_VPU0["executes"],
            "vpu0_epilogues": EXPECTED_VPU0["epilogues"],
            "vpu0_iterations": EXPECTED_VPU0["iterations"],
            "vpu0_read_resps": EXPECTED_VPU0["read_resps"],
            "vpu0_write_resps": EXPECTED_VPU0["write_resps"],
            "vpu1_completed_cmds": EXPECTED_VPU1["completed_cmds"],
            "vpu1_prologues": EXPECTED_VPU1["prologues"],
            "vpu1_executes": EXPECTED_VPU1["executes"],
            "vpu1_epilogues": EXPECTED_VPU1["epilogues"],
            "vpu1_iterations": EXPECTED_VPU1["iterations"],
            "vpu1_read_resps": EXPECTED_VPU1["read_resps"],
            "vpu1_write_resps": EXPECTED_VPU1["write_resps"],
            "shared_lut_requests": EXPECTED_SHARED_LUT_REQUESTS,
            "shared_lut_commands": EXPECTED_SHARED_LUT_COMMANDS,
        },
    )
    if not verified:
        return False

    return (
        snapshot["shared_lut_latency"] > 0
        and snapshot["shared_lut_completion_tick"]
        == snapshot["vpu1_lut_completion_tick"]
        and snapshot["vpu1_lut_completion_tick"]
        > snapshot["vpu0_linear_completion_tick"]
    )


builder, process, seu = build_m5_system(args)
root = builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_sync(process=process)
builder.map_vpu(vpu_id=0, process=process)
builder.map_vpu(vpu_id=1, process=process)
builder.map_spm(process=process)

exit_event = m5.simulate()
result = collect_simulation_result(builder, seu, exit_event)
emit_summary("SYSTEM_VPU_DUAL_RELEASE_GATE", result)

if verify_simulation_result(result):
    print("SYSTEM_VPU_DUAL_RELEASE_GATE_PASS")
