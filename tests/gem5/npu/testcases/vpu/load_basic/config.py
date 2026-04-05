# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import emit_summary, verify_snapshot  # noqa: E402
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0
EXPECTED_SNAPSHOT = {
    "exit_cause": EXPECTED_EXIT_CAUSE,
    "exit_code": EXPECTED_EXIT_CODE,
    "completed_cmds": 1,
    "prologues": 1,
    "executes": 1,
    "epilogues": 1,
    "iterations": 1,
    "read_resps": 2,
    "write_resps": 0,
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


def collect_simulation_result(builder, exit_event):
    vpu0 = builder.system.vpu0
    return {
        "exit_cause": exit_event.getCause(),
        "exit_code": exit_event.getCode(),
        "completed_cmds": vpu0.completedCmdCount(),
        "prologues": vpu0.prologueCount(),
        "executes": vpu0.executeCount(),
        "epilogues": vpu0.epilogueCount(),
        "iterations": vpu0.completedIterationCount(),
        "read_resps": vpu0.completedReadRespCount(),
        "write_resps": vpu0.completedWriteRespCount(),
    }


def verify_simulation_result(snapshot):
    return verify_snapshot(snapshot, EXPECTED_SNAPSHOT)


builder, process = build_m5_system(args)
builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_spm(process=process)
builder.map_vpu(vpu_id=0, process=process)

exit_event = m5.simulate()
result = collect_simulation_result(builder, exit_event)
emit_summary("VPU_LOAD_BASIC", result)

if verify_simulation_result(result):
    print("VPU_LOAD_BASIC_PASS")
