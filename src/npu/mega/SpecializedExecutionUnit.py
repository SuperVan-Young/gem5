# Copyright (c) 2026
# All rights reserved.

from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.SimObject import *


class SpecializedExecutionUnit(ClockedObject):
    type = "SpecializedExecutionUnit"
    cxx_header = "npu/mega/SpecializedExecutionUnit.hh"
    cxx_class = "gem5::SpecializedExecutionUnit"

    cpu_side = ResponsePort("CPU-side MMIO request input port")
    mem_side = VectorRequestPort("Memory-side active request ports")

    base_addr = Param.Addr(0x70000000, "SEU MMIO base address")
    macro_cmd_bytes = Param.Unsigned(16, "Macro command size in bytes")
    cmd_queue_depth = Param.Unsigned(4, "Internal command queue depth")
    num_mem_side_ports = Param.Unsigned(
        1, "Number of memory-side request ports"
    )
    debug_process_latency = Param.Latency(
        "100ns", "Fixed execution latency returned by the debug process path"
    )
    sync_enqueue_on_data_write = Param.Bool(
        False,
        "If true, writing command data area immediately enqueues the current staging buffer",
    )

    cxx_exports = [
        PyBindMethod("queueOccupancy"),
        PyBindMethod("completedCmdCount"),
        PyBindMethod("completedReadRespCount"),
        PyBindMethod("completedWriteRespCount"),
        PyBindMethod("completedIterationCount"),
        PyBindMethod("prologueCount"),
        PyBindMethod("executeCount"),
        PyBindMethod("epilogueCount"),
        PyBindMethod("maxActiveMicroOps"),
        PyBindMethod("isIssueBusy"),
        PyBindMethod("setDebugProcessLatency"),
    ]
