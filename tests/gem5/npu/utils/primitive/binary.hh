#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_BINARY_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_BINARY_HH_

#include "internal.hh"

static inline size_t
vpu_primitive_binary(uint32_t device_id, uint32_t op_code,
                     const PrimitiveTensorDesc &src0,
                     const PrimitiveTensorDesc &src1,
                     const PrimitiveTensorDesc &dst,
                     const PrimitiveBufferAssignment &buffers = {},
                     uint32_t sync_indicator = 0U,
                     uint64_t port_base = NPU_CMD_PORT_BASE)
{
    const uint64_t slices = dst.sliceCount(2U);
    const uint64_t src0_slices = src0.sliceCount(2U);
    const uint64_t src1_slices = src1.sliceCount(2U);
    if ((src0_slices != 1U && src0_slices != slices) ||
        (src1_slices != 1U && src1_slices != slices)) {
        __builtin_trap();
    }

    const PrimitiveVpu2DDesc first_src0 =
        primitiveLowerLastAxis2D(primitivePeelTo2D(src0, 0U));
    const PrimitiveVpu2DDesc first_src1 =
        primitiveLowerLastAxis2D(primitivePeelTo2D(src1, 0U));
    const PrimitiveVpu2DDesc first_dst =
        primitiveLowerLastAxis2D(primitivePeelTo2D(dst, 0U));
    const std::vector<NpuCmd> template_cmds = primitiveBinaryTemplate(
        device_id, op_code, first_src0, first_src1, first_dst, buffers);

    for (uint64_t i = 0U; i < slices; ++i) {
        const PrimitiveVpu2DDesc slice_src0 = primitiveLowerLastAxis2D(
            primitivePeelTo2D(src0, src0_slices == 1U ? 0U : i));
        const PrimitiveVpu2DDesc slice_src1 = primitiveLowerLastAxis2D(
            primitivePeelTo2D(src1, src1_slices == 1U ? 0U : i));
        const PrimitiveVpu2DDesc slice_dst =
            primitiveLowerLastAxis2D(primitivePeelTo2D(dst, i));
        std::vector<NpuCmd> commands = template_cmds;
        vpu_cmd_set_tensor(&commands[0], VPU_CMD_WORD_SRC0_ADDR,
                           &slice_src0.tensor);
        vpu_cmd_set_tensor(&commands[1], VPU_CMD_WORD_SRC0_ADDR,
                           &slice_src1.tensor);
        vpu_cmd_set_tensor(&commands[3], VPU_CMD_WORD_DST_ADDR, &slice_dst.tensor);
        vpu_cmd_set_sync_indicator(
            &commands[3], i + 1U == slices ? sync_indicator : 0U);
        primitiveLaunchCommands(commands, port_base);
    }

    return slices * template_cmds.size();
}

#endif
