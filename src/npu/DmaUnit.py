from m5.objects.SpecializedExecutionUnit import SpecializedExecutionUnit
from m5.params import *
from m5.SimObject import *


class DmaUnit(SpecializedExecutionUnit):
    type = "DmaUnit"
    cxx_header = "npu/DmaUnit.hh"
    cxx_class = "gem5::DmaUnit"

    num_banks = Param.Unsigned(2, "Number of DMA internal workspace banks")
    bank_size = Param.Unsigned(
        64 * 1024,
        "Per-bank DMA-local workspace size in bytes",
    )
    transpose_unit_latency = Param.Latency(
        "1ns", "Per-element transpose unit latency"
    )

    cxx_exports = SpecializedExecutionUnit.cxx_exports + [
        PyBindMethod("observedBatchOverlapCount")
    ]
