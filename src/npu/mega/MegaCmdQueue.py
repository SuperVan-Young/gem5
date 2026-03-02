# Copyright (c) 2026
# All rights reserved.

from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.SimObject import *


class MegaCmdQueue(ClockedObject):
    type = "MegaCmdQueue"
    cxx_header = "npu/mega/MegaCmdQueue.hh"
    cxx_class = "gem5::MegaCmdQueue"

    cpu_side = VectorResponsePort("CPU-side request input ports")

    num_input_port = Param.Unsigned(1, "Number of CPU input ports")
    mega_cmd_width = Param.Unsigned(128, "Macro command width in bits")
    cmd_queue_depth = Param.Unsigned(16, "FIFO depth in macro commands")
    base_addr = Param.Addr(0, "MegaCmdQueue MMIO base address")

    cxx_exports = [
        PyBindMethod("queueOccupancy"),
    ]
