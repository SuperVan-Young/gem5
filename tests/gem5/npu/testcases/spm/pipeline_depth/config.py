# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import (  # noqa: E402
    collect_seu_snapshot,
    emit_summary,
)
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_COMPLETED = 1
EXPECTED_READS = 256

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()


def build_m5_system(binary):
    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_spm(
        latency="10ns",
        bandwidth="1GiB/s",
        pipeline_depth=4,
    )
    builder.add_cpu(cpu_id=0)
    process = builder.set_workload(os.path.abspath(binary), cpu_id=0)
    builder.add_megacmdqueue()
    builder.add_seu(
        num_mem_side_ports=1,
        cmd_queue_depth=4,
        mem_port_outstanding_limit=4,
    )
    return builder, process


def collect_result(builder, exit_cause):
    seu = builder.components["seu"]
    spm = builder.components["spm"]
    snapshot = {"exit_cause": exit_cause}
    snapshot.update(collect_seu_snapshot(builder))
    snapshot["max_active_micro_ops"] = seu.maxActiveMicroOps()
    snapshot["max_mem_port_outstanding"] = seu.maxMemPortOutstanding()
    snapshot["spm_max_inflight"] = spm.maxInflightRequests()
    snapshot["spm_pipeline_full_retries"] = spm.pipelineFullRetries()
    return snapshot


def verify(result):
    return (
        result["exit_cause"] == EXPECTED_EXIT_CAUSE
        and result["queue_occupancy"] == 0
        and result["issue_busy"] is False
        and result["completed_cmds"] == EXPECTED_COMPLETED
        and result["prologues"] == EXPECTED_COMPLETED
        and result["executes"] == 0
        and result["epilogues"] == EXPECTED_COMPLETED
        and result["iterations"] == EXPECTED_COMPLETED
        and result["read_resps"] == EXPECTED_READS
        and result["write_resps"] == 0
        and result["max_active_micro_ops"] >= 2
        and result["max_mem_port_outstanding"] >= 2
        and result["spm_max_inflight"] >= 2
    )


builder, process = build_m5_system(args.binary)
root = builder.instantiate_root()
m5.instantiate()
builder.map_cmdq(process=process)
builder.map_seu(process=process)

exit_event = m5.simulate()
result = collect_result(builder, exit_event.getCause())
emit_summary("SPM_PIPELINE_DEPTH", result)

if verify(result):
    print("SPM_PIPELINE_DEPTH_PASS")
