# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "configs")
)

from npu_test_system import NPUTestSystemBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

cmd_width = 128
cmdq_base = 0x70000000
cmd_bytes = cmd_width // 8
expected_exit_cause = "exiting with last active thread context"
expected_final_occupancy = 2

binary = os.path.abspath(args.binary)

builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_cpu(cpu_id=0)
process = builder.set_workload(binary)
builder.add_megacmdqueue(
    num_input_port=1,
    mega_cmd_width=cmd_width,
    cmd_queue_depth=2,
    base_addr=cmdq_base,
)

builder.instantiate_root()
m5.instantiate()

builder.map_cmdq(process=process, size=2 * cmd_bytes, base_addr=cmdq_base)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
final_occupancy = builder.system.cmdq.queueOccupancy()

print(f"MEGACMDQUEUE_EXIT_CAUSE={exit_cause}")
print(f"MEGACMDQUEUE_FINAL_OCCUPANCY={final_occupancy}")

if (
    exit_cause == expected_exit_cause
    and final_occupancy == expected_final_occupancy
):
    print("MEGACMDQUEUE_TEST_PASS")
