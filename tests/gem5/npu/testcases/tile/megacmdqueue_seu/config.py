# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import (  # noqa: E402
    collect_cmdq_snapshot,
    collect_seu_snapshot,
    emit_summary,
)
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

NUM_CMDS = 6
EXPECTED_EXIT_CAUSE = "exiting with last active thread context"

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()


def build_m5_system(binary):
    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(binary)
    builder.add_megacmdqueue()
    builder.add_seu(debug_process_latency="200ns")
    return builder, process


def collect_simulation_result(builder, exit_cause):
    result = {"exit_cause": exit_cause}
    result.update(collect_cmdq_snapshot(builder))
    seu_snapshot = collect_seu_snapshot(builder)
    result["seu_queue_occupancy"] = seu_snapshot["queue_occupancy"]
    result["seu_completed_cmds"] = seu_snapshot["completed_cmds"]
    result["seu_busy"] = seu_snapshot["issue_busy"]
    return result


def verify_simulation_result(result):
    return (
        result["exit_cause"] == EXPECTED_EXIT_CAUSE
        and result["queue_occupancy"] == 0
        and result["seu_queue_occupancy"] == 0
        and not result["seu_busy"]
        and result["seu_completed_cmds"] >= NUM_CMDS
    )


binary = os.path.abspath(args.binary)
builder, process = build_m5_system(binary)
root = builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)

exit_event = m5.simulate()
result = collect_simulation_result(builder, exit_event.getCause())
emit_summary("TILE_MEGACMDQUEUE_SEU", result)

if verify_simulation_result(result):
    print("TILE_MEGACMDQUEUE_SEU_PASS")
