# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

sys.path.append(str(Path(__file__).resolve().parents[2] / "configs"))

from npu_test_system import NPUTestSystemBuilder

expected_exit_cause = "exiting with last active thread context"
expected_exit_code = 0
expected_vpu = {
    "completed": 6,
    "prologues": 8,
    "read_resps": 8,
    "executes": 8,
    "write_resps": 8,
    "epilogues": 8,
    "iterations": 8,
}

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_spm()
builder.add_cpu(cpu_id=0)
builder.set_workload(binary, cpu_id=0)
builder.add_megacmdqueue()
builder.add_vpu(vpu_id=0, num_mem_side_ports=4)
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq()
builder.map_spm()
builder.map_vpu(vpu_id=0)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()

vpu = {
    "completed": builder.system.vpu0.completedCmdCount(),
    "prologues": builder.system.vpu0.prologueCount(),
    "read_resps": builder.system.vpu0.completedReadRespCount(),
    "executes": builder.system.vpu0.executeCount(),
    "write_resps": builder.system.vpu0.completedWriteRespCount(),
    "epilogues": builder.system.vpu0.epilogueCount(),
    "iterations": builder.system.vpu0.completedIterationCount(),
}
lut_requests = builder.system.vpu0.lutRequestCount()
lut_commands = builder.system.vpu0.lutCommandCount()
linear_latency = builder.system.vpu0.lastLinearExecuteLatency()
lut_latency = builder.system.vpu0.lastLutExecuteLatency()
linear_completion_tick = builder.system.vpu0.lastLinearCompletionTick()
lut_completion_tick = builder.system.vpu0.lastLutCompletionTick()
lut_unit_requests = builder.system.vpu0.lut.requestCount()
lut_unit_commands = builder.system.vpu0.lut.commandCount()
lut_unit_latency = builder.system.vpu0.lut.lastExecuteLatency()
lut_unit_completion_tick = builder.system.vpu0.lut.lastCompletionTick()

print(f"VPU_UNARY_EXIT_CAUSE={exit_cause}")
print(f"VPU_UNARY_EXIT_CODE={exit_code}")
print(f"VPU_UNARY_COMPLETED_CMDS={vpu['completed']}")
print(f"VPU_UNARY_PROLOGUES={vpu['prologues']}")
print(f"VPU_UNARY_READ_RESPS={vpu['read_resps']}")
print(f"VPU_UNARY_EXECUTES={vpu['executes']}")
print(f"VPU_UNARY_WRITE_RESPS={vpu['write_resps']}")
print(f"VPU_UNARY_EPILOGUES={vpu['epilogues']}")
print(f"VPU_UNARY_ITERATIONS={vpu['iterations']}")
print(f"VPU_UNARY_LUT_REQUESTS={lut_requests}")
print(f"VPU_UNARY_LUT_COMMANDS={lut_commands}")
print(f"VPU_UNARY_LINEAR_EXEC_LATENCY={linear_latency}")
print(f"VPU_UNARY_LUT_EXEC_LATENCY={lut_latency}")
print(f"VPU_UNARY_LINEAR_COMPLETION_TICK={linear_completion_tick}")
print(f"VPU_UNARY_LUT_COMPLETION_TICK={lut_completion_tick}")
print(f"VPU_UNARY_LUT_UNIT_REQUESTS={lut_unit_requests}")
print(f"VPU_UNARY_LUT_UNIT_COMMANDS={lut_unit_commands}")
print(f"VPU_UNARY_LUT_UNIT_EXEC_LATENCY={lut_unit_latency}")
print(f"VPU_UNARY_LUT_UNIT_COMPLETION_TICK={lut_unit_completion_tick}")

if (
    exit_cause == expected_exit_cause
    and exit_code == expected_exit_code
    and vpu == expected_vpu
    and lut_requests == 8
    and lut_commands == 2
    and lut_latency > linear_latency
    and lut_completion_tick > linear_completion_tick
    and lut_unit_requests == lut_requests
    and lut_unit_commands == lut_commands
    and lut_unit_latency == lut_latency
    and lut_unit_completion_tick == lut_completion_tick
):
    print("VPU_UNARY_PASS")
