from m5.objects.SpecializedExecutionUnit import SpecializedExecutionUnit
from m5.objects.LutUnit import LutUnit
from m5.params import *
from m5.SimObject import *


class VpuUnit(SpecializedExecutionUnit):
    type = "VpuUnit"
    cxx_header = "npu/mega/VpuUnit.hh"
    cxx_class = "gem5::VpuUnit"

    device_id = Param.Unsigned(
        0, "Logical VPU device id used for MegaCmdQueue routing checks"
    )
    lut = Param.LutUnit(
        LutUnit(),
        "Shared LUT resource used by VSQRT/VEXP/VSOFTMAX; the default is a "
        "single-resource single-queue model",
    )

    cxx_exports = [
        PyBindMethod("lutRequestCount"),
        PyBindMethod("lutCommandCount"),
        PyBindMethod("lastLinearExecuteLatency"),
        PyBindMethod("lastLutExecuteLatency"),
        PyBindMethod("lastSoftmaxExecuteLatency"),
        PyBindMethod("lastLinearCompletionTick"),
        PyBindMethod("lastLutCompletionTick"),
        PyBindMethod("lastSoftmaxCompletionTick"),
    ]
