from m5.objects.SpecializedExecutionUnit import SpecializedExecutionUnit
from m5.params import *


class DmaUnit(SpecializedExecutionUnit):
    type = "DmaUnit"
    cxx_header = "npu/mega/DmaUnit.hh"
    cxx_class = "gem5::DmaUnit"

    buffer_size = Param.Unsigned(
        64 * 1024, "Unified flat-buffer size in bytes"
    )
    num_banks = Param.Unsigned(2, "Number of DMA internal workspace banks")
    bank_size = Param.Unsigned(
        0,
        "Per-bank workspace size in bytes; 0 keeps buffer_size as a temporary compatibility alias",
    )
    transpose_unit_latency = Param.Latency(
        "1ns", "Per-element transpose unit latency"
    )
