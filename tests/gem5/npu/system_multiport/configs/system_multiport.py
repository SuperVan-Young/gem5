# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5
from m5.objects import AddrRange

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "configs"))

from npu_test_system import NPUTestSystemBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

expected_exit_cause = "exiting with last active thread context"
expected_exit_code = 0
expected_dma_completed = 2
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
system = builder.build_base_system()
builder.add_default_physmem()
builder.add_spm()
builder.add_cpus(2)
builder.set_workloads(binary, lambda cpu_id: (cpu_id,), pid_base=300)
builder.add_megacmdqueue(num_input_port=2)
seu = builder.add_mega_seu(
    num_vpus=2,
    include_dma=True,
    vpu_num_mem_side_ports=4,
)
builder.instantiate_root()
m5.instantiate()

for cpu_id in range(2):
    process = builder.get_process(cpu_id)
    builder.map_cmdq_port(cpu_id, process=process)
    builder.map_dram(process=process)
    builder.map_spm(process=process)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()
cmdq_occupancy = system.cmdq.queueOccupancy()
dma_occupancy = seu.dma.queueOccupancy()
dma_busy = seu.dma.isIssueBusy()
dma_completed = seu.dma.completedCmdCount()
vpu0_occupancy = seu.vpu0.queueOccupancy()
vpu1_occupancy = seu.vpu1.queueOccupancy()
vpu0_busy = seu.vpu0.isIssueBusy()
vpu1_busy = seu.vpu1.isIssueBusy()
shared_lut_requests = seu.lut.requestCount()
shared_lut_commands = seu.lut.commandCount()
linear_completion_tick = seu.vpu0.lastLinearCompletionTick()
softmax_completion_tick = seu.vpu1.lastSoftmaxCompletionTick()
linear_latency = seu.vpu0.lastLinearExecuteLatency()
softmax_latency = seu.vpu1.lastSoftmaxExecuteLatency()
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

print(f"SYSTEM_MULTIPORT_EXIT_CAUSE={exit_cause}")
print(f"SYSTEM_MULTIPORT_EXIT_CODE={exit_code}")
print(f"SYSTEM_MULTIPORT_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"SYSTEM_MULTIPORT_DMA_OCCUPANCY={dma_occupancy}")
print(f"SYSTEM_MULTIPORT_DMA_BUSY={dma_busy}")
print(f"SYSTEM_MULTIPORT_DMA_COMPLETED={dma_completed}")
print(f"SYSTEM_MULTIPORT_VPU0_OCCUPANCY={vpu0_occupancy}")
print(f"SYSTEM_MULTIPORT_VPU1_OCCUPANCY={vpu1_occupancy}")
print(f"SYSTEM_MULTIPORT_VPU0_BUSY={vpu0_busy}")
print(f"SYSTEM_MULTIPORT_VPU1_BUSY={vpu1_busy}")
print(f"SYSTEM_MULTIPORT_VPU0_COMPLETED={vpu0['completed']}")
print(f"SYSTEM_MULTIPORT_VPU0_PROLOGUES={vpu0['prologues']}")
print(f"SYSTEM_MULTIPORT_VPU0_EXECUTES={vpu0['executes']}")
print(f"SYSTEM_MULTIPORT_VPU0_EPILOGUES={vpu0['epilogues']}")
print(f"SYSTEM_MULTIPORT_VPU0_ITERATIONS={vpu0['iterations']}")
print(f"SYSTEM_MULTIPORT_VPU0_READ_RESPS={vpu0['read_resps']}")
print(f"SYSTEM_MULTIPORT_VPU0_WRITE_RESPS={vpu0['write_resps']}")
print(f"SYSTEM_MULTIPORT_VPU1_COMPLETED={vpu1['completed']}")
print(f"SYSTEM_MULTIPORT_VPU1_PROLOGUES={vpu1['prologues']}")
print(f"SYSTEM_MULTIPORT_VPU1_EXECUTES={vpu1['executes']}")
print(f"SYSTEM_MULTIPORT_VPU1_EPILOGUES={vpu1['epilogues']}")
print(f"SYSTEM_MULTIPORT_VPU1_ITERATIONS={vpu1['iterations']}")
print(f"SYSTEM_MULTIPORT_VPU1_READ_RESPS={vpu1['read_resps']}")
print(f"SYSTEM_MULTIPORT_VPU1_WRITE_RESPS={vpu1['write_resps']}")
print(f"SYSTEM_MULTIPORT_SHARED_LUT_REQUESTS={shared_lut_requests}")
print(f"SYSTEM_MULTIPORT_SHARED_LUT_COMMANDS={shared_lut_commands}")
print(f"SYSTEM_MULTIPORT_LINEAR_EXEC_LATENCY={linear_latency}")
print(f"SYSTEM_MULTIPORT_SOFTMAX_EXEC_LATENCY={softmax_latency}")
print(f"SYSTEM_MULTIPORT_LINEAR_COMPLETION_TICK={linear_completion_tick}")
print(f"SYSTEM_MULTIPORT_SOFTMAX_COMPLETION_TICK={softmax_completion_tick}")

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
    print("SYSTEM_MULTIPORT_CONFIG_PASS")
