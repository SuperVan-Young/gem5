#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_SOFTMAX_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_SOFTMAX_HH_

#include <array>

#include "internal.hh"

static inline size_t
vpu_primitive_softmax_f32(uint32_t device_id, const PrimitiveTensorDesc &src,
                          const PrimitiveTensorDesc &dst,
                          uint32_t scratch_base_slot,
                          uint32_t local_buffer_base = 0U,
                          const PrimitiveSyncDesc &sync = {},
                          uint64_t port_base = NPU_CMD_PORT_BASE)
{
    (void)local_buffer_base;
    const uint64_t slices = src.sliceCount(2U);
    if (dst.sliceCount(2U) != slices) {
        __builtin_trap();
    }

    const PrimitiveTensorDesc src2d = primitivePeelTo2D(src, 0U);
    const PrimitiveTensorDesc dst2d = primitivePeelTo2D(dst, 0U);
    uint32_t slot = scratch_base_slot;
    const PrimitiveTensorDesc max = primitivePackedReduceLike(slot, src2d);
    slot += primitiveTensorSlotSpan(max);
    const PrimitiveTensorDesc shifted = primitivePackedLike(slot, src2d);
    slot += primitiveTensorSlotSpan(shifted);
    const PrimitiveTensorDesc exp = primitivePackedLike(slot, src2d);
    slot += primitiveTensorSlotSpan(exp);
    const PrimitiveTensorDesc sum = primitivePackedReduceLike(slot, src2d);

    const PrimitiveVpu2DDesc src_vpu = primitiveLowerLastAxis2D(src2d);
    const PrimitiveVpu2DDesc dst_vpu = primitiveLowerLastAxis2D(dst2d);
    const PrimitiveVpu2DDesc max_vpu = primitiveLowerLastAxis2D(max);
    const PrimitiveVpu2DDesc shifted_vpu = primitiveLowerLastAxis2D(shifted);
    const PrimitiveVpu2DDesc exp_vpu = primitiveLowerLastAxis2D(exp);
    const PrimitiveVpu2DDesc sum_vpu = primitiveLowerLastAxis2D(sum);
    const uint32_t max_slot_span = primitiveTensorSlotSpan(max);
    const uint32_t shifted_slot_span = primitiveTensorSlotSpan(shifted);
    const uint32_t exp_slot_span = primitiveTensorSlotSpan(exp);
    const uint32_t src_slice_stride_bytes =
        src.rank() > 2U ? (src.strideElems[src.rank() - 3U] * src.elemBytes()) : 0U;
    const uint32_t dst_slice_stride_bytes =
        dst.rank() > 2U ? (dst.strideElems[dst.rank() - 3U] * dst.elemBytes()) : 0U;
    const uint32_t scratch_slice_stride_slots =
        max_slot_span + shifted_slot_span + exp_slot_span +
        primitiveTensorSlotSpan(sum);
    const uint32_t scratch_slice_stride_bytes =
        scratch_slice_stride_slots * VPU_LOCAL_SLOT_STRIDE;

    std::array<NpuCmd, 5> commands = {
        primitiveBuildReduceExecCmd(
            device_id, VPU_OP_VREDUCE_MAX, src_vpu, max_vpu),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VSUB, src_vpu, max_vpu, shifted_vpu),
        primitiveBuildUnaryExecCmd(
            device_id, VPU_OP_VEXP, shifted_vpu, exp_vpu, 0U),
        primitiveBuildReduceExecCmd(
            device_id, VPU_OP_VREDUCE_SUM, exp_vpu, sum_vpu),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VDIV, exp_vpu, sum_vpu, dst_vpu),
    };

    for (uint64_t i = 0U; i < slices; ++i) {
        const bool is_first_slice = i == 0U;
        const bool is_last_slice = i + 1U == slices;
        const uint32_t src_addr =
            src_vpu.tensor.addr + (i * src_slice_stride_bytes);
        const uint32_t dst_addr =
            dst_vpu.tensor.addr + (i * dst_slice_stride_bytes);
        const uint32_t scratch_addr =
            max_vpu.tensor.addr + (i * scratch_slice_stride_bytes);
        const uint32_t shifted_addr =
            scratch_addr + (max_slot_span * VPU_LOCAL_SLOT_STRIDE);
        const uint32_t exp_addr =
            shifted_addr + (shifted_slot_span * VPU_LOCAL_SLOT_STRIDE);
        const uint32_t sum_addr =
            exp_addr + (exp_slot_span * VPU_LOCAL_SLOT_STRIDE);

        if (!is_first_slice) {
            primitivePatchTensorAddr(
                &commands[0], VPU_CMD_WORD_SRC0_ADDR, src_addr);
            primitivePatchTensorAddr(
                &commands[0], VPU_CMD_WORD_DST_ADDR, scratch_addr);
        }
        commands[0].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(
                &commands[1], VPU_CMD_WORD_SRC0_ADDR, src_addr);
            primitivePatchTensorAddr(
                &commands[1], VPU_CMD_WORD_SRC1_ADDR, scratch_addr);
            primitivePatchTensorAddr(
                &commands[1], VPU_CMD_WORD_DST_ADDR, shifted_addr);
        }
        commands[1].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(
                &commands[2], VPU_CMD_WORD_SRC0_ADDR, shifted_addr);
            primitivePatchTensorAddr(
                &commands[2], VPU_CMD_WORD_DST_ADDR, exp_addr);
        }
        commands[2].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(
                &commands[3], VPU_CMD_WORD_SRC0_ADDR, exp_addr);
            primitivePatchTensorAddr(
                &commands[3], VPU_CMD_WORD_DST_ADDR, sum_addr);
        }
        commands[3].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(
                &commands[4], VPU_CMD_WORD_SRC0_ADDR, exp_addr);
            primitivePatchTensorAddr(
                &commands[4], VPU_CMD_WORD_SRC1_ADDR, sum_addr);
            primitivePatchTensorAddr(
                &commands[4], VPU_CMD_WORD_DST_ADDR, dst_addr);
        }
        if (is_last_slice) {
            primitiveSetSync(&commands[4], sync);
        }
        commands[4].launchCmdAt(port_base);
    }

    return slices * commands.size();
}

#endif
