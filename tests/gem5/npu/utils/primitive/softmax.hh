#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_SOFTMAX_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_SOFTMAX_HH_

#include "binary.hh"
#include "reduce.hh"
#include "unary.hh"

static inline size_t
vpu_primitive_softmax_f32(uint32_t device_id, const PrimitiveTensorDesc &src,
                          const PrimitiveTensorDesc &dst,
                          uint32_t scratch_base_slot,
                          uint32_t local_buffer_base = 0U,
                          uint32_t sync_indicator = 0U,
                          uint64_t port_base = NPU_CMD_PORT_BASE)
{
    const uint64_t slices = src.sliceCount(2U);
    size_t command_count = 0U;

    for (uint64_t i = 0U; i < slices; ++i) {
        const PrimitiveTensorDesc src2d = primitivePeelTo2D(src, i);
        const PrimitiveTensorDesc dst2d = primitivePeelTo2D(dst, i);
        uint32_t slot = scratch_base_slot;
        const PrimitiveTensorDesc max = primitivePackedReduceLike(slot, src2d);
        slot += primitiveTensorSlotSpan(max);
        const PrimitiveTensorDesc shifted = primitivePackedLike(slot, src2d);
        slot += primitiveTensorSlotSpan(shifted);
        const PrimitiveTensorDesc exp = primitivePackedLike(slot, src2d);
        slot += primitiveTensorSlotSpan(exp);
        const PrimitiveTensorDesc sum = primitivePackedReduceLike(slot, src2d);

        command_count += vpu_primitive_reduce(
            device_id, VPU_OP_VREDUCE_MAX, src2d, max,
            {local_buffer_base + 0U, local_buffer_base + 1U,
             local_buffer_base + 0U},
            0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VSUB, src2d, max, shifted,
            {local_buffer_base + 2U, local_buffer_base + 3U,
             local_buffer_base + 1U},
            0U, port_base);
        command_count += vpu_primitive_unary(
            device_id, VPU_OP_VEXP, shifted, exp,
            {local_buffer_base + 4U, local_buffer_base + 5U,
             local_buffer_base + 2U},
            0U, 0U, port_base);
        command_count += vpu_primitive_reduce(
            device_id, VPU_OP_VREDUCE_SUM, exp, sum,
            {local_buffer_base + 6U, local_buffer_base + 7U,
             local_buffer_base + 3U},
            0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VDIV, exp, sum, dst2d,
            {local_buffer_base + 8U, local_buffer_base + 9U,
             local_buffer_base + 4U},
            i + 1U == slices ? sync_indicator : 0U, port_base);
    }

    return command_count;
}

#endif
