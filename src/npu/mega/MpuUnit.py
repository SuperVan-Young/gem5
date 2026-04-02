from m5.objects.SpecializedExecutionUnit import SpecializedExecutionUnit
from m5.params import *
from m5.SimObject import *


class MpuUnit(SpecializedExecutionUnit):
    type = "MpuUnit"
    cxx_header = "npu/mega/MpuUnit.hh"
    cxx_class = "gem5::MpuUnit"

    array_rows = Param.Unsigned(4, "Number of rows in the matrix array")
    array_cols = Param.Unsigned(4, "Number of columns in the matrix array")
    array_k_depth = Param.Unsigned(4, "K dimension depth of the matrix array")
    load_base_latency = Param.Latency(
        "1ns", "Fixed MPU-local latency for fine-grained load handling"
    )
    store_base_latency = Param.Latency(
        "1ns", "Fixed MPU-local latency for fine-grained store handling"
    )
    array_fill_latency = Param.Latency(
        "1ns", "Array fill latency component for matrix compute"
    )
    array_steady_per_k = Param.Latency(
        "1ns", "Per-k steady-state latency component for matrix compute"
    )
    array_drain_latency = Param.Latency(
        "1ns", "Array drain latency component for matrix compute"
    )

    cxx_exports = SpecializedExecutionUnit.cxx_exports + [
        PyBindMethod("loadCmdCount"),
        PyBindMethod("computeCmdCount"),
        PyBindMethod("storeCmdCount"),
        PyBindMethod("tensorLoopCmdCount"),
        PyBindMethod("matmulCount"),
        PyBindMethod("matmulAccCount"),
        PyBindMethod("slotA0Valid"),
        PyBindMethod("slotA1Valid"),
        PyBindMethod("slotB0Valid"),
        PyBindMethod("slotB1Valid"),
        PyBindMethod("slotC0Valid"),
        PyBindMethod("slotC1Valid"),
        PyBindMethod("slotC0Dirty"),
        PyBindMethod("slotC1Dirty"),
        PyBindMethod("tensorLoopExpandedTiles"),
    ]
