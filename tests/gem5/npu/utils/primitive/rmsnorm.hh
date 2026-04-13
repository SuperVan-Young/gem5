#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_RMSNORM_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_RMSNORM_HH_

#include <array>

#include "../npu_mem.hh"
#include "internal.hh"

static inline volatile uint32_t *
primitiveWordPtr(uint32_t addr)
{
    return reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
}

static inline void
primitiveStoreScalar(uint32_t slot, float value)
{
    npu_spm_clear_slot(slot);
    npu_spm_slot_word_ptr_default(slot)[0] = primitiveFloatToBits(value);
}

static inline PrimitiveTensorDesc
primitiveReplicatedWeightSlice(uint32_t slot, const PrimitiveTensorDesc &weight,
                               uint32_t rows)
{
    PrimitiveTensorDesc out;
    out.baseAddr = 0x60000000U + (slot * VPU_LOCAL_SLOT_STRIDE);
    out.dataType = VPU_DATA_F32;
    out.shape = {rows, weight.dim(0)};
    out.strideElems = {1U, rows};
    volatile uint32_t *dst = primitiveWordPtr(out.baseAddr);
    const volatile uint32_t *src = primitiveWordPtr(weight.baseAddr);
    for (uint32_t row = 0U; row < rows; ++row) {
        for (uint32_t col = 0U; col < weight.dim(0); ++col) {
            dst[col * rows + row] = src[col];
        }
    }
    return out;
}

static inline size_t
vpu_primitive_rmsnorm_f32(uint32_t device_id, const PrimitiveTensorDesc &src,
                          const PrimitiveTensorDesc &weight,
                          const PrimitiveTensorDesc &dst,
                          uint32_t scratch_base_slot,
                          uint32_t local_buffer_base = 0U,
                          float epsilon = 0.0f,
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
    const PrimitiveTensorDesc square = primitivePackedLike(slot, src2d);
    slot += primitiveTensorSlotSpan(square);
    const PrimitiveTensorDesc sum = primitivePackedReduceLike(slot, src2d);
    slot += primitiveTensorSlotSpan(sum);
    const PrimitiveTensorDesc eps = primitivePackedScalar(slot);
    slot += primitiveTensorSlotSpan(eps);
    const PrimitiveTensorDesc one = primitivePackedScalar(slot);
    slot += primitiveTensorSlotSpan(one);
    const PrimitiveTensorDesc inv_rms = primitivePackedReduceLike(slot, src2d);
    slot += primitiveTensorSlotSpan(inv_rms);
    const PrimitiveTensorDesc normalized = primitivePackedLike(slot, src2d);
    slot += primitiveTensorSlotSpan(normalized);
    const PrimitiveTensorDesc weight2d =
        primitiveReplicatedWeightSlice(slot, weight, src2d.dim(0));

    primitiveStoreScalar((eps.baseAddr - 0x60000000U) / VPU_LOCAL_SLOT_STRIDE,
                         epsilon);
    primitiveStoreScalar((one.baseAddr - 0x60000000U) / VPU_LOCAL_SLOT_STRIDE,
                         1.0f);

    const PrimitiveVpu2DDesc src_vpu = primitiveLowerLastAxis2D(src2d);
    const PrimitiveVpu2DDesc dst_vpu = primitiveLowerLastAxis2D(dst2d);
    const PrimitiveVpu2DDesc square_vpu = primitiveLowerLastAxis2D(square);
    const PrimitiveVpu2DDesc sum_vpu = primitiveLowerLastAxis2D(sum);
    const PrimitiveVpu2DDesc eps_vpu = primitiveLowerLastAxis2D(eps);
    const PrimitiveVpu2DDesc one_vpu = primitiveLowerLastAxis2D(one);
    const PrimitiveVpu2DDesc inv_rms_vpu = primitiveLowerLastAxis2D(inv_rms);
    const PrimitiveVpu2DDesc normalized_vpu =
        primitiveLowerLastAxis2D(normalized);
    const PrimitiveVpu2DDesc weight_vpu = primitiveLowerLastAxis2D(weight2d);
    const uint32_t inv_cols_scale = primitiveFloatToBits(
        1.0f / static_cast<float>(src2d.dim(src2d.rank() - 1U)));
    const uint32_t square_slot_span = primitiveTensorSlotSpan(square);
    const uint32_t sum_slot_span = primitiveTensorSlotSpan(sum);
    const uint32_t eps_slot_span = primitiveTensorSlotSpan(eps);
    const uint32_t one_slot_span = primitiveTensorSlotSpan(one);
    const uint32_t inv_rms_slot_span = primitiveTensorSlotSpan(inv_rms);
    const uint32_t src_slice_stride_bytes =
        src.rank() > 2U ? (src.strideElems[src.rank() - 3U] * src.elemBytes()) : 0U;
    const uint32_t dst_slice_stride_bytes =
        dst.rank() > 2U ? (dst.strideElems[dst.rank() - 3U] * dst.elemBytes()) : 0U;
    const uint32_t scratch_slice_stride_bytes =
        (square_slot_span + sum_slot_span + eps_slot_span + one_slot_span +
         inv_rms_slot_span + primitiveTensorSlotSpan(normalized)) *
        VPU_LOCAL_SLOT_STRIDE;

    std::array<NpuCmd, 8> commands = {
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VMUL, src_vpu, src_vpu, square_vpu),
        primitiveBuildReduceExecCmd(
            device_id, VPU_OP_VREDUCE_SUM, square_vpu, sum_vpu),
        primitiveBuildUnaryExecCmd(
            device_id, VPU_OP_VSCALE, sum_vpu, sum_vpu, inv_cols_scale),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VADD, sum_vpu, eps_vpu, sum_vpu),
        primitiveBuildUnaryExecCmd(
            device_id, VPU_OP_VSQRT, sum_vpu, inv_rms_vpu, 0U),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VDIV, one_vpu, inv_rms_vpu, inv_rms_vpu),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VMUL, src_vpu, inv_rms_vpu, normalized_vpu),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VMUL, normalized_vpu, weight_vpu, dst_vpu),
    };

    for (uint64_t i = 0U; i < slices; ++i) {
        const bool is_first_slice = i == 0U;
        const bool is_last_slice = i + 1U == slices;
        const uint32_t src_addr =
            src_vpu.tensor.addr + (i * src_slice_stride_bytes);
        const uint32_t dst_addr =
            dst_vpu.tensor.addr + (i * dst_slice_stride_bytes);
        const uint32_t square_addr =
            square_vpu.tensor.addr + (i * scratch_slice_stride_bytes);
        const uint32_t sum_addr =
            square_addr + (square_slot_span * VPU_LOCAL_SLOT_STRIDE);
        const uint32_t inv_rms_addr = sum_addr +
            (sum_slot_span + eps_slot_span + one_slot_span) *
                VPU_LOCAL_SLOT_STRIDE;
        const uint32_t normalized_addr =
            inv_rms_addr + (inv_rms_slot_span * VPU_LOCAL_SLOT_STRIDE);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[0], VPU_CMD_WORD_SRC0_ADDR,
                                     src_addr);
            primitivePatchTensorAddr(&commands[0], VPU_CMD_WORD_SRC1_ADDR,
                                     src_addr);
            primitivePatchTensorAddr(&commands[0], VPU_CMD_WORD_DST_ADDR,
                                     square_addr);
        }
        commands[0].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[1], VPU_CMD_WORD_SRC0_ADDR,
                                     square_addr);
            primitivePatchTensorAddr(&commands[1], VPU_CMD_WORD_DST_ADDR,
                                     sum_addr);
        }
        commands[1].launchCmdAt(port_base);

        commands[2].launchCmdAt(port_base);
        commands[3].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[4], VPU_CMD_WORD_SRC0_ADDR,
                                     sum_addr);
            primitivePatchTensorAddr(&commands[4], VPU_CMD_WORD_DST_ADDR,
                                     inv_rms_addr);
        }
        commands[4].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[5], VPU_CMD_WORD_SRC1_ADDR,
                                     inv_rms_addr);
            primitivePatchTensorAddr(&commands[5], VPU_CMD_WORD_DST_ADDR,
                                     inv_rms_addr);
        }
        commands[5].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[6], VPU_CMD_WORD_SRC0_ADDR,
                                     src_addr);
            primitivePatchTensorAddr(&commands[6], VPU_CMD_WORD_SRC1_ADDR,
                                     inv_rms_addr);
            primitivePatchTensorAddr(&commands[6], VPU_CMD_WORD_DST_ADDR,
                                     normalized_addr);
        }
        commands[6].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[7], VPU_CMD_WORD_SRC0_ADDR,
                                     normalized_addr);
            primitivePatchTensorAddr(&commands[7], VPU_CMD_WORD_DST_ADDR,
                                     dst_addr);
        }
        if (is_last_slice) {
            primitiveSetSync(&commands[7], sync);
        }
        commands[7].launchCmdAt(port_base);
    }

    return slices * commands.size();
}

#endif
