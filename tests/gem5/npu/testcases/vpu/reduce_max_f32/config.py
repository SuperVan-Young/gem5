# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0
EXPECTED_RESULT = {
    "exit_cause": EXPECTED_EXIT_CAUSE,
    "exit_code": EXPECTED_EXIT_CODE,
    "queue_occupancy": 0,
    "issue_busy": False,
    "completed_cmds": 3,
    "prologues": 3,
    "executes": 1,
    "epilogues": 3,
    "iterations": 3,
    "read_resps": 1,
    "write_resps": 1,
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
    return builder, process


def collect_simulation_result(builder, exit_cause, exit_code):
    vpu = builder.system.vpu0
    return {
        "exit_cause": exit_cause,
        "exit_code": exit_code,
        "queue_occupancy": vpu.queueOccupancy(),
        "issue_busy": vpu.isIssueBusy(),
        "completed_cmds": vpu.completedCmdCount(),
        "prologues": vpu.prologueCount(),
        "executes": vpu.executeCount(),
        "epilogues": vpu.epilogueCount(),
        "iterations": vpu.completedIterationCount(),
        "read_resps": vpu.completedReadRespCount(),
        "write_resps": vpu.completedWriteRespCount(),
    }


def verify_simulation_result(snapshot):
    return snapshot == EXPECTED_RESULT


builder, process = build_m5_system(args)
builder.instantiate_root()
m5.instantiate()
builder.map_cmdq()
builder.map_spm()
builder.map_vpu(vpu_id=0)

exit_event = m5.simulate()
result = collect_simulation_result(
    builder, exit_event.getCause(), exit_event.getCode()
)

for key, value in result.items():
    print(f"VPU_REDUCE_MAX_F32_{key.upper()}={value}")

if verify_simulation_result(result):
    print("VPU_REDUCE_MAX_F32_PASS")
