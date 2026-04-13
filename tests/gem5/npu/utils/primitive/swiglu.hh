#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_SWIGLU_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_SWIGLU_HH_

#include <array>

#include "../npu_mem.hh"
#include "internal.hh"

static inline size_t
vpu_primitive_swiglu_f32(uint32_t device_id, const PrimitiveTensorDesc &gate,
                         const PrimitiveTensorDesc &value,
                         const PrimitiveTensorDesc &dst,
                         uint32_t scratch_base_slot,
                         uint32_t local_buffer_base = 0U,
                         const PrimitiveSyncDesc &sync = {},
                         uint64_t port_base = NPU_CMD_PORT_BASE)
{
    (void)local_buffer_base;
    const uint64_t slices = gate.sliceCount(2U);
    if (value.sliceCount(2U) != slices || dst.sliceCount(2U) != slices) {
        __builtin_trap();
    }
    const PrimitiveTensorDesc gate2d = primitivePeelTo2D(gate, 0U);
    const PrimitiveTensorDesc value2d = primitivePeelTo2D(value, 0U);
    const PrimitiveTensorDesc dst2d = primitivePeelTo2D(dst, 0U);
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

    const PrimitiveVpu2DDesc gate_vpu = primitiveLowerLastAxis2D(gate2d);
    const PrimitiveVpu2DDesc value_vpu = primitiveLowerLastAxis2D(value2d);
    const PrimitiveVpu2DDesc dst_vpu = primitiveLowerLastAxis2D(dst2d);
    const PrimitiveVpu2DDesc neg_vpu = primitiveLowerLastAxis2D(neg);
    const PrimitiveVpu2DDesc exp_vpu = primitiveLowerLastAxis2D(exp);
    const PrimitiveVpu2DDesc denom_vpu = primitiveLowerLastAxis2D(denom);
    const PrimitiveVpu2DDesc sigmoid_vpu = primitiveLowerLastAxis2D(sigmoid);
    const PrimitiveVpu2DDesc one_vpu = primitiveLowerLastAxis2D(one);
    const uint32_t neg_slot_span = primitiveTensorSlotSpan(neg);
    const uint32_t exp_slot_span = primitiveTensorSlotSpan(exp);
    const uint32_t denom_slot_span = primitiveTensorSlotSpan(denom);
    const uint32_t gate_slice_stride_bytes =
        gate.rank() > 2U ? (gate.strideElems[gate.rank() - 3U] * gate.elemBytes()) :
                           0U;
    const uint32_t value_slice_stride_bytes =
        value.rank() > 2U ? (value.strideElems[value.rank() - 3U] *
                             value.elemBytes()) :
                            0U;
    const uint32_t dst_slice_stride_bytes =
        dst.rank() > 2U ? (dst.strideElems[dst.rank() - 3U] * dst.elemBytes()) : 0U;
    const uint32_t scratch_slice_stride_bytes =
        (neg_slot_span + exp_slot_span + denom_slot_span +
         primitiveTensorSlotSpan(sigmoid)) *
        VPU_LOCAL_SLOT_STRIDE;

    std::array<NpuCmd, 6> commands = {
        primitiveBuildUnaryExecCmd(
            device_id, VPU_OP_VSCALE, gate_vpu, neg_vpu,
            primitiveFloatToBits(-1.0f)),
        primitiveBuildUnaryExecCmd(
            device_id, VPU_OP_VEXP, neg_vpu, exp_vpu, 0U),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VADD, exp_vpu, one_vpu, denom_vpu),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VDIV, one_vpu, denom_vpu, sigmoid_vpu),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VMUL, gate_vpu, sigmoid_vpu, sigmoid_vpu),
        primitiveBuildBinaryExecCmd(
            device_id, VPU_OP_VMUL, sigmoid_vpu, value_vpu, dst_vpu),
    };

    for (uint64_t i = 0U; i < slices; ++i) {
        const bool is_first_slice = i == 0U;
        const bool is_last_slice = i + 1U == slices;
        const uint32_t gate_addr =
            gate_vpu.tensor.addr + (i * gate_slice_stride_bytes);
        const uint32_t value_addr =
            value_vpu.tensor.addr + (i * value_slice_stride_bytes);
        const uint32_t dst_addr =
            dst_vpu.tensor.addr + (i * dst_slice_stride_bytes);
        const uint32_t neg_addr =
            neg_vpu.tensor.addr + (i * scratch_slice_stride_bytes);
        const uint32_t exp_addr =
            neg_addr + (neg_slot_span * VPU_LOCAL_SLOT_STRIDE);
        const uint32_t denom_addr =
            exp_addr + (exp_slot_span * VPU_LOCAL_SLOT_STRIDE);
        const uint32_t sigmoid_addr =
            denom_addr + (denom_slot_span * VPU_LOCAL_SLOT_STRIDE);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[0], VPU_CMD_WORD_SRC0_ADDR,
                                     gate_addr);
            primitivePatchTensorAddr(&commands[0], VPU_CMD_WORD_DST_ADDR,
                                     neg_addr);
        }
        commands[0].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[1], VPU_CMD_WORD_SRC0_ADDR,
                                     neg_addr);
            primitivePatchTensorAddr(&commands[1], VPU_CMD_WORD_DST_ADDR,
                                     exp_addr);
        }
        commands[1].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[2], VPU_CMD_WORD_SRC0_ADDR,
                                     exp_addr);
            primitivePatchTensorAddr(&commands[2], VPU_CMD_WORD_DST_ADDR,
                                     denom_addr);
        }
        commands[2].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[3], VPU_CMD_WORD_SRC1_ADDR,
                                     denom_addr);
            primitivePatchTensorAddr(&commands[3], VPU_CMD_WORD_DST_ADDR,
                                     sigmoid_addr);
        }
        commands[3].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[4], VPU_CMD_WORD_SRC0_ADDR,
                                     gate_addr);
            primitivePatchTensorAddr(&commands[4], VPU_CMD_WORD_SRC1_ADDR,
                                     sigmoid_addr);
            primitivePatchTensorAddr(&commands[4], VPU_CMD_WORD_DST_ADDR,
                                     sigmoid_addr);
        }
        commands[4].launchCmdAt(port_base);

        if (!is_first_slice) {
            primitivePatchTensorAddr(&commands[5], VPU_CMD_WORD_SRC0_ADDR,
                                     sigmoid_addr);
            primitivePatchTensorAddr(&commands[5], VPU_CMD_WORD_SRC1_ADDR,
                                     value_addr);
            primitivePatchTensorAddr(&commands[5], VPU_CMD_WORD_DST_ADDR,
                                     dst_addr);
        }
        if (is_last_slice) {
            primitiveSetSync(&commands[5], sync);
        }
        commands[5].launchCmdAt(port_base);
    }

    return slices * commands.size();
}

#endif
