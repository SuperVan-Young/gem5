# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys

import m5

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))

from npu_test_system import NPUTestSystemBuilder  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

expected_exit_cause = "exiting with last active thread context"

builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_cpu(cpu_id=0)

binary = os.path.abspath(args.binary)
builder.set_workload(binary)

builder.add_megacmdqueue()
builder.add_seu()

builder.instantiate_root(full_system=False)
m5.instantiate()

builder.map_cmdq()

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
cmdq_occupancy = builder.components["cmdq"].queueOccupancy()
seu_occupancy = builder.components["seu"].queueOccupancy()
seu_completed = builder.components["seu"].completedCmdCount()

print(f"SIT_EXIT_CAUSE={exit_cause}")
print(f"SIT_EXIT_OK={int(exit_cause == expected_exit_cause)}")
print(f"SIT_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"SIT_SEU_OCCUPANCY={seu_occupancy}")
print(f"SIT_SEU_COMPLETED={seu_completed}")

if (
    exit_cause == expected_exit_cause
    and cmdq_occupancy == 0
    and seu_occupancy == 0
    and seu_completed == 2
):
    print("TEST_PASS")
