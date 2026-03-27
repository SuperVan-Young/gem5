from m5.params import *
from m5.SimObject import *


class LutUnit(SimObject):
    type = "LutUnit"
    cxx_header = "npu/mega/LutUnit.hh"
    cxx_class = "gem5::LutUnit"

    range_reduction_latency = Param.Latency(
        "20ns",
        "Per-request latency for nonlinear range reduction in the current "
        "single-resource LUT model",
    )
    lookup_latency = Param.Latency(
        "30ns",
        "Per-request latency for LUT table lookup in the current "
        "single-resource LUT model",
    )
    interpolation_latency = Param.Latency(
        "20ns",
        "Per-request latency for LUT interpolation in the current "
        "single-resource LUT model",
    )
    normalize_latency = Param.Latency(
        "20ns",
        "Per-request latency for nonlinear post-processing or normalization "
        "in the current single-resource LUT model",
    )
    table_entries = Param.Unsigned(
        257,
        "Number of entries in the sqrt/exp lookup tables used by the current "
        "approximation model",
    )

    cxx_exports = [
        PyBindMethod("requestCount"),
        PyBindMethod("commandCount"),
        PyBindMethod("lastExecuteLatency"),
        PyBindMethod("lastCompletionTick"),
        PyBindMethod("lastSoftmaxExecuteLatency"),
        PyBindMethod("lastSoftmaxCompletionTick"),
    ]
