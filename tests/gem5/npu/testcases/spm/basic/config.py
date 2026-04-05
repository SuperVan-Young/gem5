# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_common import emit_summary  # noqa: E402
from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()


def build_m5_system(binary):
    builder = NPUTestSystemBuilder()
    builder.build_base_system()
    builder.add_default_physmem()
    builder.add_cpu()
    process = builder.set_workload(binary)
    builder.add_spm()
    return builder, process


def collect_simulation_result(exit_event):
    return {
        "exit_cause": exit_event.getCause(),
        "exit_code": exit_event.getCode(),
    }


def verify_simulation_result(result):
    return (
        result["exit_cause"] == EXPECTED_EXIT_CAUSE
        and result["exit_code"] == EXPECTED_EXIT_CODE
    )


binary = os.path.abspath(args.binary)
builder, process = build_m5_system(binary)
root = builder.instantiate_root()
m5.instantiate()
builder.map_spm(process=process)

exit_event = m5.simulate()
result = collect_simulation_result(exit_event)
emit_summary("SPM_BASIC", result)

if verify_simulation_result(result):
    print("SPM_BASIC_PASS")
