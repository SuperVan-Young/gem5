# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import (  # noqa: E402
    build_dma_functional_system,
    verify_snapshot,
)

SCENARIO_NAME = "staged_transpose_basic"
EXPECTED_SNAPSHOT = {
    "exit_cause": "exiting with last active thread context",
    "exit_code": 0,
    "scenario": SCENARIO_NAME,
    "cmds": 3,
    "reads": 2,
    "writes": 1,
    "overlap": 0,
    "queue": 0,
    "cmdq": 0,
    "busy": 0,
}

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

binary = os.path.abspath(args.binary)
builder, process = build_dma_functional_system(binary, scenario=SCENARIO_NAME)
builder.instantiate_root()
system = builder.system

m5.instantiate()
builder.map_cmdq(process=process)
builder.map_spm(process=process)
builder.map_dram(process=process)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()

print(f"DMA_EXIT_CAUSE={exit_cause}")
print(f"DMA_EXIT_CODE={exit_code}")
print(f"DMA_SCENARIO={SCENARIO_NAME}")

snapshot = {
    "exit_cause": exit_cause,
    "exit_code": exit_code,
    "scenario": SCENARIO_NAME,
    "cmds": system.dma.completedCmdCount(),
    "reads": system.dma.completedReadRespCount(),
    "writes": system.dma.completedWriteRespCount(),
    "overlap": system.dma.observedBatchOverlapCount(),
    "queue": system.dma.queueOccupancy(),
    "cmdq": system.cmdq.queueOccupancy(),
    "busy": int(system.dma.isIssueBusy()),
    "active": system.dma.maxActiveMicroOps(),
}

print(
    "DMA_SUMMARY "
    f"scenario={snapshot['scenario']} "
    f"cmds={snapshot['cmds']} "
    f"reads={snapshot['reads']} "
    f"writes={snapshot['writes']} "
    f"overlap={snapshot['overlap']} "
    f"queue={snapshot['queue']} "
    f"cmdq={snapshot['cmdq']} "
    f"busy={snapshot['busy']} "
    f"active={snapshot['active']}"
)

if verify_snapshot(
    {key: snapshot[key] for key in EXPECTED_SNAPSHOT},
    EXPECTED_SNAPSHOT,
):
    print(f"DMA_SCENARIO_PASS={SCENARIO_NAME}")
