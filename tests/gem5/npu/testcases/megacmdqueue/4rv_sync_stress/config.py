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

NUM_CPUS = 4
EXPECTED_EXIT_CAUSE = "exiting with last active thread context"

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--rounds", type=int, default=64)
args = parser.parse_args()


def build_m5_system(args):
    assert args.rounds > 0

    binary = os.path.abspath(args.binary)
    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_cpus(NUM_CPUS)
    builder.set_workloads(
        binary,
        lambda cpu_id: (cpu_id, args.rounds),
        pid_base=100,
    )
    builder.add_megacmdqueue(num_input_port=NUM_CPUS)
    builder.add_seu()
    return builder, builder.get_processes()


def collect_simulation_result(builder, exit_cause, expected_total):
    snapshot = {"exit_cause": exit_cause, "expected_total": expected_total}
    snapshot.update(collect_cmdq_snapshot(builder))
    seu_snapshot = collect_seu_snapshot(builder)
    snapshot["seu_queue_occupancy"] = seu_snapshot["queue_occupancy"]
    snapshot["seu_completed_cmds"] = seu_snapshot["completed_cmds"]
    return snapshot


def verify_simulation_result(snapshot):
    return (
        snapshot["exit_cause"] == EXPECTED_EXIT_CAUSE
        and snapshot["queue_occupancy"] == 0
        and snapshot["seu_queue_occupancy"] == 0
        and snapshot["seu_completed_cmds"] == snapshot["expected_total"]
    )


builder, processes = build_m5_system(args)
root = builder.instantiate_root(full_system=False)
m5.instantiate()

for cpu_id, process in enumerate(processes):
    builder.map_cmdq_port(cpu_id, process=process)
    builder.map_sync(process=process, base_addr=builder.addr_map.sync_base)

exit_event = m5.simulate()
result = collect_simulation_result(
    builder,
    exit_event.getCause(),
    NUM_CPUS * args.rounds,
)
emit_summary("MEGACMDQUEUE_4RV", result)

if verify_simulation_result(result):
    print("MEGACMDQUEUE_4RV_SYNC_STRESS_PASS")
