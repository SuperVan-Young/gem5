# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from os.path import dirname, join as joinpath

import m5

sys.path.insert(0, joinpath(dirname(__file__), "..", "..", "configs"))

from npu_test_system import NPUTestSystemBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

expected_exit_cause = "exiting with last active thread context"

binary = os.path.abspath(args.binary)

builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_cpu()
builder.set_workload(binary)
builder.add_megacmdqueue()
builder.add_seu()
builder.instantiate_root()
m5.instantiate()

process = builder.get_process(0)
builder.map_cmdq(process=process)
builder.map_sync(process=process)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
cmdq_occupancy = builder.system.cmdq.queueOccupancy()
seu_occupancy = builder.system.seu.queueOccupancy()

print(f"SIT_EXIT_CAUSE={exit_cause}")
print(f"SIT_EXIT_OK={int(exit_cause == expected_exit_cause)}")
print(f"SIT_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"SIT_SEU_OCCUPANCY={seu_occupancy}")

if (
    exit_cause == expected_exit_cause
    and cmdq_occupancy == 0
    and seu_occupancy == 0
):
    print("TEST_PASS")
