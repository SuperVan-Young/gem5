# Copyright (c) 2026
# All rights reserved.

from pathlib import Path
import runpy

runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "mpu"
        / "configs"
        / "mpu_proto.py"
    ),
    run_name="__main__",
)
