# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys

import m5

npu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if npu_root not in sys.path:
    sys.path.insert(0, npu_root)

from configs.npu_test_system import NPUTestSystemBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

num_cmds = 6

expected_exit_cause = "exiting with last active thread context"

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_cpu(cpu_id=0)
process = builder.set_workload(binary)
builder.add_megacmdqueue()
builder.add_seu(debug_process_latency="200ns")
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq(process=process)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
cmdq_occupancy = builder.system.cmdq.queueOccupancy()
seu_occupancy = builder.system.seu.queueOccupancy()
seu_completed = builder.system.seu.completedCmdCount()
seu_busy = builder.system.seu.isIssueBusy()

print(f"TILE_EXIT_CAUSE={exit_cause}")
print(f"TILE_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"TILE_SEU_OCCUPANCY={seu_occupancy}")
print(f"TILE_SEU_COMPLETED={seu_completed}")
print(f"TILE_SEU_BUSY={seu_busy}")

if (
    exit_cause == expected_exit_cause
    and cmdq_occupancy == 0
    and seu_occupancy == 0
    and not seu_busy
    and seu_completed >= num_cmds
):
    print("TEST_PASS")
