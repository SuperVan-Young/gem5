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
    "completed": 2,
    "prologues": 2,
    "executes": 2,
    "epilogues": 2,
    "iterations": 2,
    "read_resps": 2,
    "write_resps": 2,
}

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_spm()
builder.add_cpu(cpu_id=0)
process = builder.set_workload(binary, cpu_id=0)
builder.add_megacmdqueue()
seu = builder.add_mega_seu(
    num_vpus=1,
    include_dma=False,
    vpu_num_mem_side_ports=3,
)
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq(process=process)
builder.map_sync(process=process)
builder.map_vpu(vpu_id=0, process=process)
builder.map_spm(process=process)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()
cmdq_occupancy = builder.system.cmdq.queueOccupancy()
vpu_occupancy = seu.vpu0.queueOccupancy()
vpu_busy = seu.vpu0.isIssueBusy()
lut_requests = seu.vpu0.lutRequestCount()
lut_commands = seu.vpu0.lutCommandCount()
linear_latency = seu.vpu0.lastLinearExecuteLatency()
lut_latency = seu.vpu0.lastLutExecuteLatency()
linear_completion_tick = seu.vpu0.lastLinearCompletionTick()
lut_completion_tick = seu.vpu0.lastLutCompletionTick()
vpu = {
    "completed": seu.vpu0.completedCmdCount(),
    "prologues": seu.vpu0.prologueCount(),
    "executes": seu.vpu0.executeCount(),
    "epilogues": seu.vpu0.epilogueCount(),
    "iterations": seu.vpu0.completedIterationCount(),
    "read_resps": seu.vpu0.completedReadRespCount(),
    "write_resps": seu.vpu0.completedWriteRespCount(),
}

print(f"SYSTEM_BASIC_EXIT_CAUSE={exit_cause}")
print(f"SYSTEM_BASIC_EXIT_CODE={exit_code}")
print(f"SYSTEM_BASIC_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"SYSTEM_BASIC_VPU_OCCUPANCY={vpu_occupancy}")
print(f"SYSTEM_BASIC_VPU_BUSY={vpu_busy}")
print(f"SYSTEM_BASIC_VPU_COMPLETED_CMDS={vpu['completed']}")
print(f"SYSTEM_BASIC_VPU_PROLOGUES={vpu['prologues']}")
print(f"SYSTEM_BASIC_VPU_EXECUTES={vpu['executes']}")
print(f"SYSTEM_BASIC_VPU_EPILOGUES={vpu['epilogues']}")
print(f"SYSTEM_BASIC_VPU_ITERATIONS={vpu['iterations']}")
print(f"SYSTEM_BASIC_VPU_READ_RESPS={vpu['read_resps']}")
print(f"SYSTEM_BASIC_VPU_WRITE_RESPS={vpu['write_resps']}")
print(f"SYSTEM_BASIC_LUT_REQUESTS={lut_requests}")
print(f"SYSTEM_BASIC_LUT_COMMANDS={lut_commands}")
print(f"SYSTEM_BASIC_LINEAR_EXEC_LATENCY={linear_latency}")
print(f"SYSTEM_BASIC_LUT_EXEC_LATENCY={lut_latency}")
print(f"SYSTEM_BASIC_LINEAR_COMPLETION_TICK={linear_completion_tick}")
print(f"SYSTEM_BASIC_LUT_COMPLETION_TICK={lut_completion_tick}")

if (
    exit_cause == expected_exit_cause
    and exit_code == expected_exit_code
    and cmdq_occupancy == 0
    and vpu_occupancy == 0
    and not vpu_busy
    and vpu == expected_vpu
    and lut_requests == 4
    and lut_commands == 1
    and lut_latency > linear_latency
    and lut_completion_tick > linear_completion_tick
):
    print("SYSTEM_BASIC_TEST_PASS")
