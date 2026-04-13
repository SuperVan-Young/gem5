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
builder.add_spm(size=8 * 1024 * 1024)
builder.add_cpu(cpu_id=0)
process = builder.set_workload(binary, cpu_id=0)
builder.add_megacmdqueue()
builder.add_vpu(
    vpu_id=0,
    num_mem_side_ports=32,
    input_buffer_count=40,
    output_buffer_count=32,
    local_buffer_stride=128 * 128 * 4,
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
vpu = builder.system.vpu0
snapshot = {
    "exit_cause": exit_event.getCause(),
    "exit_code": exit_event.getCode(),
    "completed_cmds": vpu.completedCmdCount(),
    "lut_commands": vpu.lutCommandCount(),
    "lut_requests": vpu.lutRequestCount(),
    "linear_exec_latency": vpu.lastLinearExecuteLatency(),
    "lut_exec_latency": vpu.lastLutExecuteLatency(),
    "linear_completion_tick": vpu.lastLinearCompletionTick(),
    "lut_completion_tick": vpu.lastLutCompletionTick(),
}

for key, value in snapshot.items():
    print(f"LLM_PRIMITIVES_F32_{key.upper()}={value}")

if (
    snapshot["exit_cause"] == EXPECTED_EXIT_CAUSE
    and snapshot["exit_code"] == EXPECTED_EXIT_CODE
    and snapshot["completed_cmds"] > 0
    and snapshot["lut_commands"] > 0
    and snapshot["lut_requests"] > 0
    and snapshot["linear_exec_latency"] > 0
    and snapshot["lut_exec_latency"] > 0
    and snapshot["lut_completion_tick"] > 0
):
    print("LLM_PRIMITIVES_F32_CONFIG_PASS")
