#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_UNARY_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_UNARY_HH_

#include "internal.hh"

static inline size_t
vpu_primitive_unary(uint32_t device_id, uint32_t op_code,
                    const PrimitiveTensorDesc &src,
                    const PrimitiveTensorDesc &dst,
                    const PrimitiveBufferAssignment &buffers = {},
                    uint32_t sync_indicator = 0U, uint32_t extra0 = 0U,
                    uint64_t port_base = NPU_CMD_PORT_BASE)
{
    const uint64_t slices = src.sliceCount(2U);
    if (dst.sliceCount(2U) != slices) {
        __builtin_trap();
    }

    const PrimitiveVpu2DDesc first_src =
        primitiveLowerLastAxis2D(primitivePeelTo2D(src, 0U));
    const PrimitiveVpu2DDesc first_dst =
        primitiveLowerLastAxis2D(primitivePeelTo2D(dst, 0U));
    const std::vector<NpuCmd> template_cmds =
        primitiveUnaryTemplate(device_id, op_code, first_src, first_dst, buffers,
                               extra0);

    for (uint64_t i = 0U; i < slices; ++i) {
        const PrimitiveVpu2DDesc slice_src =
            primitiveLowerLastAxis2D(primitivePeelTo2D(src, i));
        const PrimitiveVpu2DDesc slice_dst =
            primitiveLowerLastAxis2D(primitivePeelTo2D(dst, i));
        std::vector<NpuCmd> commands = template_cmds;
        vpu_cmd_set_tensor(&commands[0], VPU_CMD_WORD_SRC0_ADDR, &slice_src.tensor);
        vpu_cmd_set_tensor(&commands[2], VPU_CMD_WORD_DST_ADDR, &slice_dst.tensor);
        vpu_cmd_set_sync_indicator(
            &commands[2], i + 1U == slices ? sync_indicator : 0U);
        primitiveLaunchCommands(commands, port_base);
    }

    return slices * template_cmds.size();
}

#endif
