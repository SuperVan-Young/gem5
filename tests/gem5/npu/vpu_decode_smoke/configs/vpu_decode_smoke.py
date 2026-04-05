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
    "read_resps": 3,
    "executes": 2,
    "write_resps": 2,
    "epilogues": 2,
    "iterations": 2,
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

print(f"VPU_DECODE_EXIT_CAUSE={exit_cause}")
print(f"VPU_DECODE_EXIT_CODE={exit_code}")
print(f"VPU_DECODE_COMPLETED_CMDS={vpu['completed']}")
print(f"VPU_DECODE_PROLOGUES={vpu['prologues']}")
print(f"VPU_DECODE_READ_RESPS={vpu['read_resps']}")
print(f"VPU_DECODE_EXECUTES={vpu['executes']}")
print(f"VPU_DECODE_WRITE_RESPS={vpu['write_resps']}")
print(f"VPU_DECODE_EPILOGUES={vpu['epilogues']}")
print(f"VPU_DECODE_ITERATIONS={vpu['iterations']}")

if (
    exit_cause == expected_exit_cause
    and exit_code == expected_exit_code
    and vpu == expected_vpu
):
    print("VPU_DECODE_SMOKE_PASS")
