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
    load_bandwidth_bytes_per_cycle = Param.Unsigned(
        16, "Modeled MPU-local load bandwidth in bytes per cycle"
    )
    store_bandwidth_bytes_per_cycle = Param.Unsigned(
        16, "Modeled MPU-local store bandwidth in bytes per cycle"
    )
    c_read_base_latency = Param.Latency(
        "1ns", "Fixed latency for local C read in MATMUL_ACC"
    )
    c_write_base_latency = Param.Latency(
        "1ns", "Fixed latency for local C write in MATMUL_ACC"
    )
    local_bank_count = Param.Unsigned(2, "Number of modeled local banks")
    local_bank_granularity_bytes = Param.Unsigned(
        4, "Granularity of local-bank arbitration in bytes"
    )
    local_bank_service_cycles = Param.Unsigned(
        1, "Cycles required to serve one local-bank granule"
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
        PyBindMethod("tensorLoopExpandedAccTiles"),
        PyBindMethod("totalInternalLoads"),
        PyBindMethod("totalInternalComputes"),
        PyBindMethod("totalInternalStores"),
        PyBindMethod("totalTiles"),
        PyBindMethod("totalAccTiles"),
        PyBindMethod("stallCyclesWaitingForSPM"),
        PyBindMethod("stallCyclesWaitingForSlot"),
        PyBindMethod("observedTotalLatency"),
        PyBindMethod("partialSumSpillCount"),
        PyBindMethod("partialSumReloadCount"),
    ]
