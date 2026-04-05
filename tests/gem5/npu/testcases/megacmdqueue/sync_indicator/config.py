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

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()


def build_m5_system(args):
    binary = os.path.abspath(args.binary)
    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_cpu()
    process = builder.set_workload(binary)
    builder.add_megacmdqueue()
    builder.add_seu()
    return builder, process


def collect_simulation_result(builder, exit_cause):
    snapshot = {"exit_cause": exit_cause}
    snapshot.update(collect_cmdq_snapshot(builder))
    snapshot["seu_queue_occupancy"] = collect_seu_snapshot(builder)[
        "queue_occupancy"
    ]
    return snapshot


def verify_simulation_result(snapshot):
    return (
        snapshot["exit_cause"] == EXPECTED_EXIT_CAUSE
        and snapshot["queue_occupancy"] == 0
        and snapshot["seu_queue_occupancy"] == 0
    )


builder, process = build_m5_system(args)
root = builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_sync(process=process)

exit_event = m5.simulate()
result = collect_simulation_result(builder, exit_event.getCause())
emit_summary("MEGACMDQUEUE_SYNC", result)

if verify_simulation_result(result):
    print("MEGACMDQUEUE_SYNC_INDICATOR_PASS")
