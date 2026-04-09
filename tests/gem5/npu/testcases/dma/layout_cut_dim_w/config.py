# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5
from m5.objects import (
    AddrRange,
    DmaUnit,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import (  # noqa: E402
    emit_summary,
    verify_snapshot,
)
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

case_name = Path(__file__).resolve().parent.name
summary_prefix = f"DMA_{case_name.upper()}"
pass_marker = f"DMA_{case_name.upper()}_PASS"


def attach_dma(builder, bank_size):
    dma = DmaUnit(
        base_addr=builder.addr_map.dma_base,
        macro_cmd_bytes=64,
        cmd_queue_depth=8,
        num_mem_side_ports=2,
        sync_enqueue_on_data_write=True,
        bank_size=bank_size,
    )
    dma.cpu_side = builder.system.membus.mem_side_ports
    for _ in range(2):
        dma.mem_side = builder.system.membus.cpu_side_ports
    builder.system.dma = dma
    builder.components["dma"] = dma
    return dma


def build_m5_system(args):
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
    process = builder.set_workload(binary, cpu_id=0)
    builder.add_megacmdqueue()
    attach_dma(
        builder,
        32 if case_name == "bank_size_forces_batching" else 4096,
    )
    return builder, process


def collect_simulation_result(builder, exit_event):
    dma = builder.system.dma
    snapshot = {
        "exit_cause": exit_event.getCause(),
        "exit_code": exit_event.getCode(),
        "completed_cmds": dma.completedCmdCount(),
        "queue_occupancy": dma.queueOccupancy(),
        "issue_busy": dma.isIssueBusy(),
    }
    if case_name == "bank_size_forces_batching":
        snapshot["completed_iterations"] = dma.completedIterationCount()
    return snapshot


def verify_simulation_result(snapshot):
    expected = {
        "exit_cause": EXPECTED_EXIT_CAUSE,
        "exit_code": EXPECTED_EXIT_CODE,
        "completed_cmds": 1,
        "queue_occupancy": 0,
        "issue_busy": False,
    }
    if case_name == "bank_size_forces_batching":
        expected["completed_iterations"] = 4
    return verify_snapshot(snapshot, expected)


builder, process = build_m5_system(args)
builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_spm(process=process)
builder.map_dram(process=process)

exit_event = m5.simulate()
result = collect_simulation_result(builder, exit_event)
emit_summary(summary_prefix, result)

if verify_simulation_result(result):
    print(pass_marker)
