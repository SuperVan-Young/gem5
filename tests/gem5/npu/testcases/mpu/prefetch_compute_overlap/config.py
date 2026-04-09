# Copyright (c) 2026
# All rights reserved.

import runpy
from pathlib import Path

LEGACY_CONFIG = (
    Path(__file__).resolve().parents[3] / "mpu" / "configs" / "mpu_proto.py"
)

runpy.run_path(str(LEGACY_CONFIG), run_name="__main__")
