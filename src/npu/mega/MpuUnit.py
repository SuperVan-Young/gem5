from m5.objects.SpecializedExecutionUnit import SpecializedExecutionUnit
from m5.params import *
from m5.SimObject import *


class MpuUnit(SpecializedExecutionUnit):
    type = "MpuUnit"
    cxx_header = "npu/mega/MpuUnit.hh"
    cxx_class = "gem5::MpuUnit"

    array_dim = Param.Unsigned(8, "Square MAC array dimension")
    a_buffer_capacity_bytes = Param.Unsigned(
        4096, "Capacity of each A buffer in bytes"
    )
    b_buffer_capacity_bytes = Param.Unsigned(
        4096, "Capacity of each B buffer in bytes"
    )
    c_buffer_capacity_bytes = Param.Unsigned(
        4096, "Capacity of each C buffer in bytes"
    )
    mem_uop_queue_depth = Param.Unsigned(8, "Memory micro-op queue depth")
    exec_uop_queue_depth = Param.Unsigned(8, "Execution micro-op queue depth")
    drain_uop_queue_depth = Param.Unsigned(8, "Drain micro-op queue depth")
    mvin_request_latency = Param.Latency(
        "1ns", "Latency carried by the mvin execute stage"
    )
    mvout_request_latency = Param.Latency(
        "1ns", "Latency carried by the mvout execute stage"
    )
    load_latency_base = Param.Latency("1ns", "Base latency for load commands")
    drain_latency_base = Param.Latency(
        "1ns", "Base latency for drain commands"
    )

    cxx_exports = SpecializedExecutionUnit.cxx_exports + [
        PyBindMethod("macroFifoOccupancy"),
        PyBindMethod("memUopQueueOccupancy"),
        PyBindMethod("execUopQueueOccupancy"),
        PyBindMethod("drainUopQueueOccupancy"),
        PyBindMethod("mvinCmdCount"),
        PyBindMethod("loadCmdCount"),
        PyBindMethod("computeCmdCount"),
        PyBindMethod("drainCmdCount"),
        PyBindMethod("mvoutCmdCount"),
        PyBindMethod("totalABytesIn"),
        PyBindMethod("totalBBytesIn"),
        PyBindMethod("totalCBytesOut"),
        PyBindMethod("totalOutputElementsDrained"),
        PyBindMethod("totalMacOps"),
        PyBindMethod("busyCycles"),
        PyBindMethod("idleCycles"),
        PyBindMethod("stallCyclesWaitingForSpm"),
        PyBindMethod("stallCyclesBufferHazard"),
        PyBindMethod("stallCyclesOutputStorageUnavailable"),
        PyBindMethod("stallCyclesDrainDestBusy"),
        PyBindMethod("lastComputeLatencyCycles"),
        PyBindMethod("lastCommandLatencyCycles"),
        PyBindMethod("currentCmdKind"),
        PyBindMethod("aBufferState"),
        PyBindMethod("bBufferState"),
        PyBindMethod("cBufferState"),
        PyBindMethod("outputStorageStateCode"),
        PyBindMethod("loadedAIndex"),
        PyBindMethod("loadedBIndex"),
        PyBindMethod("scoreboardLoadedAReady"),
        PyBindMethod("scoreboardLoadedBReady"),
        PyBindMethod("scoreboardOutputReady"),
    ]
