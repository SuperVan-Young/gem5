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

cmd_width = 128
cmd_bytes = cmd_width // 8
queue_depth = 4
expected_exit_cause = "exiting with last active thread context"

binary = os.path.abspath(args.binary)

builder = NPUTestSystemBuilder()
system = builder.build_base_system()
builder.add_default_physmem()
builder.add_cpu(cpu_id=0)
builder.set_workload(binary)
builder.add_megacmdqueue(
    num_input_port=1,
    mega_cmd_width=cmd_width,
    cmd_queue_depth=queue_depth,
    range_addr=builder.addr_map.cmdq_range_base,
    num_sync_indicator=256,
)
builder.add_seu(
    base_addr=builder.addr_map.seu_base,
    macro_cmd_bytes=cmd_bytes,
    cmd_queue_depth=queue_depth,
    debug_process_latency="50ns",
    sync_enqueue_on_data_write=True,
)
root = builder.instantiate_root(full_system=False)
m5.instantiate()

process = builder.get_process(0)
builder.map_cmdq(process=process, base_addr=builder.addr_map.cmdq_base)
builder.map_sync(process=process, base_addr=builder.addr_map.sync_base)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
cmdq_occupancy = system.cmdq.queueOccupancy()
seu_occupancy = system.seu.queueOccupancy()
seu_completed = system.seu.completedCmdCount()

print(f"SIT_EXIT_CAUSE={exit_cause}")
print(f"SIT_EXIT_OK={int(exit_cause == expected_exit_cause)}")
print(f"SIT_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"SIT_SEU_OCCUPANCY={seu_occupancy}")
print(f"SIT_SEU_COMPLETED={seu_completed}")

if (
    exit_cause == expected_exit_cause
    and cmdq_occupancy == 0
    and seu_occupancy == 0
    and seu_completed >= 1
):
    print("TEST_PASS")
