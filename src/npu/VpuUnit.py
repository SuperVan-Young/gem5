from m5.objects.LutUnit import LutUnit
from m5.objects.SpecializedExecutionUnit import SpecializedExecutionUnit
from m5.params import *
from m5.SimObject import *


class VpuUnit(SpecializedExecutionUnit):
    type = "VpuUnit"
    cxx_header = "npu/VpuUnit.hh"
    cxx_class = "gem5::VpuUnit"

    device_id = Param.Unsigned(
        0, "Logical VPU device id used for MegaCmdQueue routing checks"
    )
    lut = Param.LutUnit(
        LutUnit(),
        "Shared LUT resource used by VSQRT/VEXP; the default is a "
        "single-resource single-queue model",
    )
    num_input_ports = Param.Unsigned(
        2, "Number of logical local input ports visible to compute commands"
    )
    num_output_ports = Param.Unsigned(
        1, "Number of logical local output ports visible to compute commands"
    )
    input_buffer_count = Param.Unsigned(
        2, "Number of local input buffers per logical port"
    )
    output_buffer_count = Param.Unsigned(
        2, "Number of local output buffers per logical port"
    )
    local_input_base = Param.Addr(
        0x80000000, "Base tag used to encode local input-buffer addresses"
    )
    local_output_base = Param.Addr(
        0x81000000, "Base tag used to encode local output-buffer addresses"
    )
    local_buffer_stride = Param.Unsigned(
        0x40,
        "Stride of one logical local-buffer slot in the encoded address map",
    )
    dlen_bytes = Param.Unsigned(
        4,
        "Logical dlen in bytes; the innermost layout dimension times dtype "
        "width must equal this value",
    )
    dlen_group_size = Param.Unsigned(
        16,
        "Number of dlen chunks modeled by one VPU internal pipeline group",
    )
    int8_cycles_per_dlen = Param.Cycles(
        1, "Linear VPU cycles needed to process one dlen chunk of int8/uint8"
    )
    int16_cycles_per_dlen = Param.Cycles(
        1, "Linear VPU cycles needed to process one dlen chunk of int16/uint16"
    )
    int32_cycles_per_dlen = Param.Cycles(
        2, "Linear VPU cycles needed to process one dlen chunk of int32/uint32"
    )
    float16_cycles_per_dlen = Param.Cycles(
        2, "Linear VPU cycles needed to process one dlen chunk of float16"
    )
    float32_cycles_per_dlen = Param.Cycles(
        4, "Linear VPU cycles needed to process one dlen chunk of float32"
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
