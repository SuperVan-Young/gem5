# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5
from m5.objects import AddrRange

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument(
    "--scenario",
    default="spm_only",
    choices=("spm_only", "copy_back"),
)
args = parser.parse_args()

sys.path.append(str(Path(__file__).resolve().parents[2] / "configs"))

from npu_test_system import NPUTestSystemBuilder

expected_exit_cause = "exiting with last active thread context"
expected_exit_code = 0
expected_dma_completed = 2 if args.scenario == "copy_back" else 1
expected_vpu0 = {
    "completed": 1,
    "prologues": 1,
    "executes": 1,
    "epilogues": 1,
    "iterations": 1,
    "read_resps": 1,
    "write_resps": 1,
}
expected_vpu1 = {
    "completed": 1,
    "prologues": 1,
    "executes": 1,
    "epilogues": 1,
    "iterations": 1,
    "read_resps": 1,
    "write_resps": 1,
}

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder(
    mem_ranges=[
        AddrRange(0, size=0x60000000),
        AddrRange(0x60000000, size=64 * 1024),
    ]
)
builder.build_base_system()
builder.add_default_physmem()
builder.add_spm()
builder.add_cpu(cpu_id=0)
process = builder.set_workload(binary, argv=[args.scenario], cpu_id=0)
builder.add_megacmdqueue()
seu = builder.add_mega_seu(
    num_vpus=2,
    include_dma=True,
    vpu_num_mem_side_ports=4,
)
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq(process=process)
builder.map_sync(process=process)
builder.map_dram(process=process)
builder.map_spm(process=process)
builder.map_vpu(vpu_id=0, process=process)
builder.map_vpu(vpu_id=1, process=process)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()
cmdq_occupancy = builder.system.cmdq.queueOccupancy()
dma_occupancy = seu.dma.queueOccupancy()
dma_busy = seu.dma.isIssueBusy()
dma_completed = seu.dma.completedCmdCount()
vpu0_occupancy = seu.vpu0.queueOccupancy()
vpu1_occupancy = seu.vpu1.queueOccupancy()
vpu0_busy = seu.vpu0.isIssueBusy()
vpu1_busy = seu.vpu1.isIssueBusy()
shared_lut_requests = seu.lut.requestCount()
shared_lut_commands = seu.lut.commandCount()
linear_latency = seu.vpu0.lastLinearExecuteLatency()
softmax_latency = seu.vpu1.lastSoftmaxExecuteLatency()
linear_completion_tick = seu.vpu0.lastLinearCompletionTick()
softmax_completion_tick = seu.vpu1.lastSoftmaxCompletionTick()
vpu0 = {
    "completed": seu.vpu0.completedCmdCount(),
    "prologues": seu.vpu0.prologueCount(),
    "executes": seu.vpu0.executeCount(),
    "epilogues": seu.vpu0.epilogueCount(),
    "iterations": seu.vpu0.completedIterationCount(),
    "read_resps": seu.vpu0.completedReadRespCount(),
    "write_resps": seu.vpu0.completedWriteRespCount(),
}
vpu1 = {
    "completed": seu.vpu1.completedCmdCount(),
    "prologues": seu.vpu1.prologueCount(),
    "executes": seu.vpu1.executeCount(),
    "epilogues": seu.vpu1.epilogueCount(),
    "iterations": seu.vpu1.completedIterationCount(),
    "read_resps": seu.vpu1.completedReadRespCount(),
    "write_resps": seu.vpu1.completedWriteRespCount(),
}

print(f"SYSTEM_PIPELINE_EXIT_CAUSE={exit_cause}")
print(f"SYSTEM_PIPELINE_EXIT_CODE={exit_code}")
print(f"SYSTEM_PIPELINE_SCENARIO={args.scenario}")
print(f"SYSTEM_PIPELINE_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"SYSTEM_PIPELINE_DMA_OCCUPANCY={dma_occupancy}")
print(f"SYSTEM_PIPELINE_DMA_BUSY={dma_busy}")
print(f"SYSTEM_PIPELINE_DMA_COMPLETED={dma_completed}")
print(f"SYSTEM_PIPELINE_VPU0_OCCUPANCY={vpu0_occupancy}")
print(f"SYSTEM_PIPELINE_VPU1_OCCUPANCY={vpu1_occupancy}")
print(f"SYSTEM_PIPELINE_VPU0_BUSY={vpu0_busy}")
print(f"SYSTEM_PIPELINE_VPU1_BUSY={vpu1_busy}")
print(f"SYSTEM_PIPELINE_VPU0_COMPLETED={vpu0['completed']}")
print(f"SYSTEM_PIPELINE_VPU0_PROLOGUES={vpu0['prologues']}")
print(f"SYSTEM_PIPELINE_VPU0_EXECUTES={vpu0['executes']}")
print(f"SYSTEM_PIPELINE_VPU0_EPILOGUES={vpu0['epilogues']}")
print(f"SYSTEM_PIPELINE_VPU0_ITERATIONS={vpu0['iterations']}")
print(f"SYSTEM_PIPELINE_VPU0_READ_RESPS={vpu0['read_resps']}")
print(f"SYSTEM_PIPELINE_VPU0_WRITE_RESPS={vpu0['write_resps']}")
print(f"SYSTEM_PIPELINE_VPU1_COMPLETED={vpu1['completed']}")
print(f"SYSTEM_PIPELINE_VPU1_PROLOGUES={vpu1['prologues']}")
print(f"SYSTEM_PIPELINE_VPU1_EXECUTES={vpu1['executes']}")
print(f"SYSTEM_PIPELINE_VPU1_EPILOGUES={vpu1['epilogues']}")
print(f"SYSTEM_PIPELINE_VPU1_ITERATIONS={vpu1['iterations']}")
print(f"SYSTEM_PIPELINE_VPU1_READ_RESPS={vpu1['read_resps']}")
print(f"SYSTEM_PIPELINE_VPU1_WRITE_RESPS={vpu1['write_resps']}")
print(f"SYSTEM_PIPELINE_SHARED_LUT_REQUESTS={shared_lut_requests}")
print(f"SYSTEM_PIPELINE_SHARED_LUT_COMMANDS={shared_lut_commands}")
print(f"SYSTEM_PIPELINE_LINEAR_EXEC_LATENCY={linear_latency}")
print(f"SYSTEM_PIPELINE_SOFTMAX_EXEC_LATENCY={softmax_latency}")
print(f"SYSTEM_PIPELINE_LINEAR_COMPLETION_TICK={linear_completion_tick}")
print(f"SYSTEM_PIPELINE_SOFTMAX_COMPLETION_TICK={softmax_completion_tick}")

if (
    exit_cause == expected_exit_cause
    and exit_code == expected_exit_code
    and cmdq_occupancy == 0
    and dma_occupancy == 0
    and not dma_busy
    and dma_completed == expected_dma_completed
    and vpu0_occupancy == 0
    and vpu1_occupancy == 0
    and not vpu0_busy
    and not vpu1_busy
    and vpu0 == expected_vpu0
    and vpu1 == expected_vpu1
    and shared_lut_requests == 4
    and shared_lut_commands == 1
    and softmax_latency > linear_latency
    and softmax_completion_tick > linear_completion_tick
):
    print(f"SYSTEM_PIPELINE_CONFIG_PASS={args.scenario}")
