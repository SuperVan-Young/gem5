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

vpu0_expected = {
    "completed": 2,
    "prologues": 3,
    "read_resps": 5,
    "executes": 3,
    "write_resps": 3,
    "epilogues": 3,
    "iterations": 3,
}
vpu1_expected = {
    "completed": 2,
    "prologues": 3,
    "read_resps": 5,
    "executes": 3,
    "write_resps": 3,
    "epilogues": 3,
    "iterations": 3,
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
builder.add_vpu(vpu_id=1, num_mem_side_ports=4)
builder.instantiate_root()
m5.instantiate()

builder.map_cmdq()
builder.map_spm()
builder.map_vpu(vpu_id=0)
builder.map_vpu(vpu_id=1)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()

vpu0 = {
    "completed": builder.system.vpu0.completedCmdCount(),
    "prologues": builder.system.vpu0.prologueCount(),
    "read_resps": builder.system.vpu0.completedReadRespCount(),
    "executes": builder.system.vpu0.executeCount(),
    "write_resps": builder.system.vpu0.completedWriteRespCount(),
    "epilogues": builder.system.vpu0.epilogueCount(),
    "iterations": builder.system.vpu0.completedIterationCount(),
}
vpu1 = {
    "completed": builder.system.vpu1.completedCmdCount(),
    "prologues": builder.system.vpu1.prologueCount(),
    "read_resps": builder.system.vpu1.completedReadRespCount(),
    "executes": builder.system.vpu1.executeCount(),
    "write_resps": builder.system.vpu1.completedWriteRespCount(),
    "epilogues": builder.system.vpu1.epilogueCount(),
    "iterations": builder.system.vpu1.completedIterationCount(),
}

print(f"VPU_EXIT_CAUSE={exit_cause}")
print(f"VPU_EXIT_CODE={exit_code}")
print(f"VPU0_COMPLETED_CMDS={vpu0['completed']}")
print(f"VPU0_PROLOGUES={vpu0['prologues']}")
print(f"VPU0_READ_RESPS={vpu0['read_resps']}")
print(f"VPU0_EXECUTES={vpu0['executes']}")
print(f"VPU0_WRITE_RESPS={vpu0['write_resps']}")
print(f"VPU0_EPILOGUES={vpu0['epilogues']}")
print(f"VPU0_ITERATIONS={vpu0['iterations']}")
print(f"VPU1_COMPLETED_CMDS={vpu1['completed']}")
print(f"VPU1_PROLOGUES={vpu1['prologues']}")
print(f"VPU1_READ_RESPS={vpu1['read_resps']}")
print(f"VPU1_EXECUTES={vpu1['executes']}")
print(f"VPU1_WRITE_RESPS={vpu1['write_resps']}")
print(f"VPU1_EPILOGUES={vpu1['epilogues']}")
print(f"VPU1_ITERATIONS={vpu1['iterations']}")

if (
    exit_cause == expected_exit_cause
    and exit_code == expected_exit_code
    and vpu0 == vpu0_expected
    and vpu1 == vpu1_expected
):
    print("VPU_TEST_PASS")
