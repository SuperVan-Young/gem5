# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "configs"))

from npu_test_system import NPUTestSystemBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--rounds", type=int, default=64)
args = parser.parse_args()

num_cpus = 4
expected_exit_cause = "exiting with last active thread context"

assert args.rounds > 0
expected_total = num_cpus * args.rounds

binary = os.path.abspath(args.binary)

builder = NPUTestSystemBuilder()
system = builder.build_base_system()
builder.add_default_physmem()
builder.add_cpus(num_cpus)
builder.set_workloads(
    binary,
    lambda cpu_id: (cpu_id, args.rounds),
    pid_base=100,
)
builder.add_megacmdqueue(
    num_input_port=num_cpus,
)
builder.add_seu()
builder.instantiate_root(full_system=False)
m5.instantiate()

for cpu_id in range(num_cpus):
    process = builder.get_process(cpu_id)
    builder.map_cmdq_port(cpu_id, process=process)
    builder.map_sync(process=process, base_addr=builder.addr_map.sync_base)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
last_cmdq_occupancy = system.cmdq.queueOccupancy()
last_seu_occupancy = system.seu.queueOccupancy()
last_seu_completed = system.seu.completedCmdCount()

print(f"FOUR_RV_EXIT_CAUSE={exit_cause}")
print(f"FOUR_RV_EXIT_OK={int(exit_cause == expected_exit_cause)}")
print(f"FOUR_RV_EXPECTED_TOTAL={expected_total}")
print(f"FOUR_RV_CMDQ_OCCUPANCY={last_cmdq_occupancy}")
print(f"FOUR_RV_SEU_OCCUPANCY={last_seu_occupancy}")
print(f"FOUR_RV_SEU_COMPLETED={last_seu_completed}")

if (
    exit_cause == expected_exit_cause
    and last_cmdq_occupancy == 0
    and last_seu_occupancy == 0
    and last_seu_completed == expected_total
):
    print("4RV_TEST_PASS")
