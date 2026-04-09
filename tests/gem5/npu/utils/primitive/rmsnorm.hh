#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_RMSNORM_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_RMSNORM_HH_

#include "../npu_mem.hh"
#include "binary.hh"
#include "reduce.hh"
#include "unary.hh"

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
                          uint32_t sync_indicator = 0U,
                          uint64_t port_base = NPU_CMD_PORT_BASE)
{
    const uint64_t slices = src.sliceCount(2U);
    size_t command_count = 0U;

    for (uint64_t i = 0U; i < slices; ++i) {
        const PrimitiveTensorDesc src2d = primitivePeelTo2D(src, i);
        const PrimitiveTensorDesc dst2d = primitivePeelTo2D(dst, i);
        const PrimitiveTensorDesc weight2d = primitiveReplicatedWeightSlice(
            scratch_base_slot + 6U, weight, src2d.dim(0));
        const PrimitiveTensorDesc square =
            primitivePackedLike(scratch_base_slot + 0U, src2d);
        const PrimitiveTensorDesc sum =
            primitivePackedReduceLike(scratch_base_slot + 1U, src2d);
        const PrimitiveTensorDesc eps =
            primitivePackedScalar(scratch_base_slot + 2U);
        const PrimitiveTensorDesc one =
            primitivePackedScalar(scratch_base_slot + 3U);
        const PrimitiveTensorDesc inv_rms =
            primitivePackedReduceLike(scratch_base_slot + 4U, src2d);
        const PrimitiveTensorDesc normalized =
            primitivePackedLike(scratch_base_slot + 5U, src2d);

        primitiveStoreScalar(scratch_base_slot + 2U, epsilon);
        primitiveStoreScalar(scratch_base_slot + 3U, 1.0f);

        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VMUL, src2d, src2d, square,
            {local_buffer_base + 0U, local_buffer_base + 1U,
             local_buffer_base + 0U},
            0U, port_base);
        command_count += vpu_primitive_reduce(
            device_id, VPU_OP_VREDUCE_SUM, square, sum,
            {local_buffer_base + 2U, local_buffer_base + 3U,
             local_buffer_base + 1U},
            0U, port_base);
        command_count += vpu_primitive_unary(
            device_id, VPU_OP_VSCALE, sum, sum,
            {local_buffer_base + 4U, local_buffer_base + 5U,
             local_buffer_base + 2U},
            0U, primitiveFloatToBits(
                    1.0f / static_cast<float>(src2d.dim(src2d.rank() - 1U))),
            port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VADD, sum, eps, sum,
            {local_buffer_base + 6U, local_buffer_base + 7U,
             local_buffer_base + 3U},
            0U, port_base);
        command_count += vpu_primitive_unary(
            device_id, VPU_OP_VSQRT, sum, inv_rms,
            {local_buffer_base + 8U, local_buffer_base + 9U,
             local_buffer_base + 4U},
            0U, 0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VDIV, one, inv_rms, inv_rms,
            {local_buffer_base + 10U, local_buffer_base + 11U,
             local_buffer_base + 5U},
            0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VMUL, src2d, inv_rms, normalized,
            {local_buffer_base + 12U, local_buffer_base + 13U,
             local_buffer_base + 6U},
            0U, port_base);
        command_count += vpu_primitive_binary(
            device_id, VPU_OP_VMUL, normalized, weight2d, dst2d,
            {local_buffer_base + 14U, local_buffer_base + 15U,
             local_buffer_base + 7U},
            i + 1U == slices ? sync_indicator : 0U, port_base);
    }

    return command_count;
}

#endif
