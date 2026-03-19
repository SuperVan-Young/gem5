from m5.objects.SpecializedExecutionUnit import SpecializedExecutionUnit
from m5.params import *


class DmaUnit(SpecializedExecutionUnit):
    type = "DmaUnit"
    cxx_header = "npu/mega/DmaUnit.hh"
    cxx_class = "gem5::DmaUnit"

    buffer_size = Param.Unsigned(
        64 * 1024, "Unified flat-buffer size in bytes"
    )
