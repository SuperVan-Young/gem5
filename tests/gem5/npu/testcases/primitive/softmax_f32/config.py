# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_spm()
builder.add_cpu(cpu_id=0)
builder.set_workload(binary, cpu_id=0)
builder.add_megacmdqueue()
builder.add_vpu(
    vpu_id=0,
    num_mem_side_ports=32,
    input_buffer_count=24,
    output_buffer_count=12,
    dlen_bytes=8,
    float32_cycles_per_dlen=4,
    float16_cycles_per_dlen=2,
    int32_cycles_per_dlen=2,
)
builder.instantiate_root()
m5.instantiate()
builder.map_cmdq()
builder.map_sync()
builder.map_spm()
builder.map_vpu(vpu_id=0)

exit_event = m5.simulate()
if (
    exit_event.getCause() == EXPECTED_EXIT_CAUSE
    and exit_event.getCode() == EXPECTED_EXIT_CODE
):
    print("SOFTMAX_F32_CONFIG_PASS")
