# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from pathlib import Path

import m5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_system import NPUTestSystemBuilder  # noqa: E402

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

binary = os.path.abspath(args.binary)
builder = NPUTestSystemBuilder()
builder.build_base_system()
builder.add_default_physmem()
builder.add_cpu()
builder.set_workload(binary)
root = builder.instantiate_root()
m5.instantiate()

exit_event = m5.simulate()
if (
    exit_event.getCause() == EXPECTED_EXIT_CAUSE
    and exit_event.getCode() == EXPECTED_EXIT_CODE
):
    print("FLASH_ATTENTION_QUANTIZE_ROWWISE_F32_TO_I8_CONFIG_PASS")
