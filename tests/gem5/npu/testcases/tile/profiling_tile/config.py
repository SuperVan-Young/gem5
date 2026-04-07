# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5
from m5.objects import AddrRange

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_system import NPUTestSystemBuilder  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder(
    clock="4GHz",
    mem_ranges=[
        AddrRange(0, size=0x60000000),
        AddrRange(0x60000000, size=64 * 1024),
    ]
)
system = builder.build_base_system()
builder.add_default_physmem()
builder.add_spm()
builder.add_cpu(cpu_id=0)
builder.set_workload(binary, cpu_id=0, pid=400)
builder.add_megacmdqueue()
lut = builder.add_lut(
    range_reduction_latency="20ns",
    lookup_latency="30ns",
    interpolation_latency="20ns",
    normalize_latency="20ns",
)
dma = builder.add_dma()
vpu0 = builder.add_vpu(
    vpu_id=0,
    num_mem_side_ports=4,
    debug_process_latency="200ns",
    lut=lut,
)
vpu1 = builder.add_vpu(
    vpu_id=1,
    num_mem_side_ports=4,
    debug_process_latency="200ns",
    lut=lut,
)
builder.instantiate_root()
m5.instantiate()

process = builder.get_process(0)
builder.map_cmdq(process=process)
builder.map_dram(process=process)
builder.map_spm(process=process)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()

cmdq_occupancy = system.cmdq.queueOccupancy()
dma_completed = dma.completedCmdCount()
vpu0_completed = vpu0.completedCmdCount()
vpu1_completed = vpu1.completedCmdCount()

print(f"PROFILE_TILE_EXIT_CAUSE={exit_cause}")
print(f"PROFILE_TILE_EXIT_CODE={exit_code}")
print(f"PROFILE_TILE_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"PROFILE_TILE_DMA_COMPLETED={dma_completed}")
print(f"PROFILE_TILE_VPU0_COMPLETED={vpu0_completed}")
print(f"PROFILE_TILE_VPU1_COMPLETED={vpu1_completed}")

if (
    exit_cause == EXPECTED_EXIT_CAUSE
    and exit_code == EXPECTED_EXIT_CODE
    and cmdq_occupancy == 0
    and dma_completed >= 2
    and vpu0_completed >= 1
    and vpu1_completed >= 1
):
    print("PROFILE_TILE_CONFIG_PASS")
