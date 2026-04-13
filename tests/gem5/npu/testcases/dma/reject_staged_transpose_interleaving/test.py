# Copyright (c) 2026
# All rights reserved.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from dma_panic_common import register_dma_panic_test  # noqa: E402

register_dma_panic_test(__file__, "reject_staged_transpose_interleaving")
