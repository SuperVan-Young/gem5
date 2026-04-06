# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import (  # noqa: E402
    collect_vpu_snapshot,
    emit_summary,
    verify_snapshot,
)
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_VPU = {
    "queue_occupancy": 0,
    "issue_busy": False,
    "completed_cmds": 4,
    "prologues": 4,
    "executes": 1,
    "epilogues": 4,
    "iterations": 4,
    "read_resps": 2,
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


def collect_simulation_result(builder, exit_cause):
    snapshot = {"exit_cause": exit_cause}
    snapshot.update(collect_vpu_snapshot(builder))
    return snapshot


def verify_simulation_result(snapshot):
    expected = {"exit_cause": EXPECTED_EXIT_CAUSE}
    expected.update(EXPECTED_VPU)
    return verify_snapshot(snapshot, expected)


builder, process = build_m5_system(args)
root = builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_spm(process=process)
builder.map_vpu(vpu_id=0, process=process)

exit_event = m5.simulate()
result = collect_simulation_result(builder, exit_event.getCause())
emit_summary("VPU_ELEMWISE_ADD_F32", result)

if verify_simulation_result(result):
    print("VPU_ELEMWISE_ADD_F32_PASS")
