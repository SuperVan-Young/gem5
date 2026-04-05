# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import (  # noqa: E402
    collect_seu_snapshot,
    emit_summary,
    verify_snapshot,
)
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_COMPLETED_CMDS = 5
EXPECTED_EXIT_CAUSE = "exiting with last active thread context"

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()


def build_m5_system(args):
    binary = os.path.abspath(args.binary)
    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(binary, cpu_id=0)
    builder.add_megacmdqueue()
    builder.add_seu()
    return builder, process


def collect_simulation_result(builder, exit_cause):
    snapshot = {"exit_cause": exit_cause}
    snapshot.update(collect_seu_snapshot(builder))
    return snapshot


def verify_simulation_result(snapshot):
    return verify_snapshot(
        snapshot,
        {
            "exit_cause": EXPECTED_EXIT_CAUSE,
            "queue_occupancy": 0,
            "issue_busy": False,
            "completed_cmds": EXPECTED_COMPLETED_CMDS,
            "prologues": EXPECTED_COMPLETED_CMDS,
            "executes": EXPECTED_COMPLETED_CMDS,
            "epilogues": EXPECTED_COMPLETED_CMDS,
            "iterations": EXPECTED_COMPLETED_CMDS,
            "read_resps": 0,
            "write_resps": 0,
        },
    )


builder, process = build_m5_system(args)
root = builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_seu(process=process)

exit_event = m5.simulate()
result = collect_simulation_result(builder, exit_event.getCause())
emit_summary("SEU_BASIC", result)

if verify_simulation_result(result):
    print("SEU_BASIC_PASS")
