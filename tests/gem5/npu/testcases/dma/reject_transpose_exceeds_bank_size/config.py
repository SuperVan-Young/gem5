# Copyright (c) 2026
# All rights reserved.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from dma_panic_common import run_dma_panic_config  # noqa: E402

run_dma_panic_config("reject_transpose_exceeds_bank_size")
