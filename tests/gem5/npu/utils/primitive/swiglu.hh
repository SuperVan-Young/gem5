#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_SWIGLU_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_SWIGLU_HH_

#include "../npu_mem.hh"
#include "binary.hh"
#include "unary.hh"

static inline size_t
vpu_primitive_swiglu_f32(uint32_t device_id, const PrimitiveTensorDesc &gate,
                         const PrimitiveTensorDesc &value,
                         const PrimitiveTensorDesc &dst,
                         uint32_t scratch_base_slot,
                         uint32_t local_buffer_base = 0U,
                         uint32_t sync_indicator = 0U,
                         uint64_t port_base = NPU_CMD_PORT_BASE)
{
    const uint64_t slices = gate.sliceCount(2U);
    size_t command_count = 0U;

    for (uint64_t i = 0U; i < slices; ++i) {
        const PrimitiveTensorDesc gate2d = primitivePeelTo2D(gate, i);
        const PrimitiveTensorDesc value2d = primitivePeelTo2D(value, i);
        const PrimitiveTensorDesc dst2d = primitivePeelTo2D(dst, i);
        uint32_t slot = scratch_base_slot;
        const PrimitiveTensorDesc neg = primitivePackedLike(slot, gate2d);
        slot += primitiveTensorSlotSpan(neg);
        const PrimitiveTensorDesc exp = primitivePackedLike(slot, gate2d);
        slot += primitiveTensorSlotSpan(exp);
        const PrimitiveTensorDesc denom = primitivePackedLike(slot, gate2d);
        slot += primitiveTensorSlotSpan(denom);
        const PrimitiveTensorDesc sigmoid = primitivePackedLike(slot, gate2d);
        slot += primitiveTensorSlotSpan(sigmoid);
        const PrimitiveTensorDesc one = primitivePackedScalar(slot);
        npu_spm_clear_slot(slot);
        npu_spm_slot_word_ptr_default(slot)[0] = primitiveFloatToBits(1.0f);

        command_count += vpu_primitive_unary(
            device_id, VPU_OP_VSCALE, gate2d, neg,
            {local_buffer_base + 0U, local_buffer_base + 1U,
             local_buffer_base + 0U},
            0U, primitiveFloatToBits(-1.0f), port_base);
        command_count += vpu_primitive_unary(
            device_id, VPU_OP_VEXP, neg, exp,
            {local_buffer_base + 2U, local_buffer_base + 3U,
             local_buffer_base + 1U},
            0U, 0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VADD, exp, one, denom,
            {local_buffer_base + 4U, local_buffer_base + 5U,
             local_buffer_base + 2U},
            0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VDIV, one, denom, sigmoid,
            {local_buffer_base + 6U, local_buffer_base + 7U,
             local_buffer_base + 3U},
            0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VMUL, gate2d, sigmoid, sigmoid,
            {local_buffer_base + 8U, local_buffer_base + 9U,
             local_buffer_base + 4U},
            0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VMUL, sigmoid, value2d, dst2d,
            {local_buffer_base + 10U, local_buffer_base + 11U,
             local_buffer_base + 5U},
            i + 1U == slices ? sync_indicator : 0U, port_base);
    }

    return command_count;
}

#endif
