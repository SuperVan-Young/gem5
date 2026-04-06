# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import collect_seu_snapshot, emit_summary  # noqa: E402
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()


def build_m5_system(binary):
    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_spm()
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(os.path.abspath(binary), cpu_id=0)
    builder.add_megacmdqueue()
    builder.add_seu(num_mem_side_ports=2, cmd_queue_depth=4)
    return builder, process


def collect_result(builder, exit_cause):
    snapshot = {"exit_cause": exit_cause}
    snapshot.update(collect_seu_snapshot(builder))
    snapshot["max_active_micro_ops"] = builder.components[
        "seu"
    ].maxActiveMicroOps()
    return snapshot


def verify(result):
    return (
        result["exit_cause"] == EXPECTED_EXIT_CAUSE
        and result["queue_occupancy"] == 0
        and result["issue_busy"] is False
        and result["completed_cmds"] == 10
        and result["prologues"] == 10
        and result["executes"] == 10
        and result["epilogues"] == 10
        and result["iterations"] == 10
        and result["read_resps"] == 7168
        and result["write_resps"] == 9216
        and result["max_active_micro_ops"] >= 2
    )


builder, process = build_m5_system(args.binary)
root = builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_seu(process=process)

exit_event = m5.simulate()
result = collect_result(builder, exit_event.getCause())
emit_summary("SEU_ISSUE_QUEUE_STRESS", result)

if verify(result):
    print("SEU_ISSUE_QUEUE_STRESS_PASS")
