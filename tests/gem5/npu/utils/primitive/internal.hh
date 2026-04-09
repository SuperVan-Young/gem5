#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_INTERNAL_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_INTERNAL_HH_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "../tensor.hh"

struct PrimitiveBufferAssignment
{
    uint32_t input0 = 0U;
    uint32_t input1 = 1U;
    uint32_t output = 0U;
};

enum
{
    PRIMITIVE_DLEN_BYTES = 8U,
};

static inline uint32_t
primitiveFloatToBits(float value)
{
    uint32_t bits = 0U;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline uint32_t
primitiveLargestPowerOfTwoDivisor(uint32_t value)
{
    uint32_t out = 1U;
    while ((value % (out << 1U)) == 0U) {
        out <<= 1U;
    }
    return out;
}

static inline PrimitiveTensorDesc
primitivePeelTo2D(const PrimitiveTensorDesc &tensor, uint64_t slice_index)
{
    if (tensor.rank() > 2U) {
        return tensor.peel(slice_index, 2U);
    }
    if (tensor.rank() == 2U) {
        return tensor;
    }
    if (tensor.rank() == 1U) {
        return tensor.asWVector2D();
    }
    return tensor.asScalar2D();
}

static inline PrimitiveVpu2DDesc
primitiveLowerLastAxis2D(const PrimitiveTensorDesc &tensor)
{
    const PrimitiveTensorDesc slice = primitivePeelTo2D(tensor, 0U);
    const uint32_t elem_bytes = slice.elemBytes();
    if (elem_bytes == 0U || (PRIMITIVE_DLEN_BYTES % elem_bytes) != 0U) {
        __builtin_trap();
    }

    const uint32_t dlen_elems = PRIMITIVE_DLEN_BYTES / elem_bytes;
    if (dlen_elems == 0U) {
        __builtin_trap();
    }

    if ((slice.dim(0) % dlen_elems) == 0U) {
        return slice.lowerToVpu2D(dlen_elems, 1U, VPU_LAYOUT_CW);
    }
    if (slice.dim(1) == 1U) {
        return slice.lowerToVpu2D(1U, dlen_elems, VPU_LAYOUT_WC);
    }
    __builtin_trap();
}

static inline PrimitiveTensorDesc
primitivePackedLike(uint32_t slot, const PrimitiveTensorDesc &tensor)
{
    PrimitiveTensorDesc out = tensor;
    out.baseAddr = 0x60000000U + (slot * VPU_LOCAL_SLOT_STRIDE);
    return out;
}

static inline PrimitiveTensorDesc
primitivePackedReduceLike(uint32_t slot, const PrimitiveTensorDesc &tensor)
{
    std::vector<uint32_t> dims = tensor.shape;
    if (dims.empty()) {
        dims.push_back(1U);
    } else {
        dims.back() = 1U;
    }
    PrimitiveTensorDesc out;
    out.baseAddr = 0x60000000U + (slot * VPU_LOCAL_SLOT_STRIDE);
    out.dataType = tensor.dataType;
    out.shape = dims;
    out.strideElems.resize(dims.size(), 1U);
    uint32_t running = 1U;
    for (size_t i = dims.size(); i > 0U; --i) {
        out.strideElems[i - 1U] = running;
        running *= dims[i - 1U];
    }
    return out;
}

static inline PrimitiveTensorDesc
primitivePackedScalar(uint32_t slot, uint32_t data_type = VPU_DATA_F32)
{
    return PrimitiveTensorDesc::denseSpm(slot, data_type, {1U});
}

static inline VpuTensorDesc
primitiveLocalTensor(uint32_t base, uint32_t buffer_index,
                     const PrimitiveVpu2DDesc &desc)
{
    return vpu_tensor_desc(vpu_local_addr(base, buffer_index), desc.tensor.shape,
                           desc.tensor.stride);
}

static inline void
primitiveLaunchCommands(const std::vector<NpuCmd> &commands, uint64_t port_base)
{
    for (const auto &cmd : commands) {
        cmd.launchCmdAt(port_base);
    }
}

static inline std::vector<NpuCmd>
primitiveUnaryTemplate(uint32_t device_id, uint32_t op_code,
                       const PrimitiveVpu2DDesc &src,
                       const PrimitiveVpu2DDesc &dst,
                       const PrimitiveBufferAssignment &buffers,
                       uint32_t extra0)
{
    std::vector<NpuCmd> commands(3);
    const VpuTensorDesc zero = vpu_tensor_desc(0U, 0U, 0U);
    const VpuTensorDesc local_src =
        primitiveLocalTensor(VPU_LOCAL_INPUT_BASE, buffers.input0, src);
    const VpuTensorDesc local_dst =
        primitiveLocalTensor(VPU_LOCAL_OUTPUT_BASE, buffers.output, dst);

    vpu_cmd_init_compute(&commands[0], device_id, VPU_OP_VLOAD, 0U, src.dataType,
                         src.dataType, src.dataType, src.wLayoutLog2,
                         src.cLayoutLog2, src.layoutOrder, &local_src, &src.tensor,
                         &zero, 0U);
    vpu_cmd_init_compute(&commands[1], device_id, op_code, 0U, dst.dataType,
                         src.dataType, src.dataType, dst.wLayoutLog2,
                         dst.cLayoutLog2, dst.layoutOrder, &local_dst, &local_src,
                         &local_src, extra0);
    vpu_cmd_init_compute(&commands[2], device_id, VPU_OP_VSTORE, 0U, dst.dataType,
                         dst.dataType, dst.dataType, dst.wLayoutLog2,
                         dst.cLayoutLog2, dst.layoutOrder, &dst.tensor, &local_dst,
                         &zero, 0U);
    return commands;
}

static inline std::vector<NpuCmd>
primitiveBinaryTemplate(uint32_t device_id, uint32_t op_code,
                        const PrimitiveVpu2DDesc &src0,
                        const PrimitiveVpu2DDesc &src1,
                        const PrimitiveVpu2DDesc &dst,
                        const PrimitiveBufferAssignment &buffers)
{
    std::vector<NpuCmd> commands(4);
    const VpuTensorDesc zero = vpu_tensor_desc(0U, 0U, 0U);
    const VpuTensorDesc local_src0 =
        primitiveLocalTensor(VPU_LOCAL_INPUT_BASE, buffers.input0, src0);
    const VpuTensorDesc local_src1 =
        primitiveLocalTensor(VPU_LOCAL_INPUT_BASE, buffers.input1, src1);
    const VpuTensorDesc local_dst =
        primitiveLocalTensor(VPU_LOCAL_OUTPUT_BASE, buffers.output, dst);

    vpu_cmd_init_compute(&commands[0], device_id, VPU_OP_VLOAD, 0U,
                         src0.dataType, src0.dataType, src0.dataType,
                         src0.wLayoutLog2, src0.cLayoutLog2, src0.layoutOrder,
                         &local_src0, &src0.tensor, &zero, 0U);
    vpu_cmd_init_compute(&commands[1], device_id, VPU_OP_VLOAD, 0U,
                         src1.dataType, src1.dataType, src1.dataType,
                         src1.wLayoutLog2, src1.cLayoutLog2, src1.layoutOrder,
                         &local_src1, &src1.tensor, &zero, 0U);
    vpu_cmd_init_compute(&commands[2], device_id, op_code, 0U, dst.dataType,
                         src0.dataType, src1.dataType, dst.wLayoutLog2,
                         dst.cLayoutLog2, dst.layoutOrder, &local_dst, &local_src0,
                         &local_src1, 0U);
    vpu_cmd_init_compute(&commands[3], device_id, VPU_OP_VSTORE, 0U, dst.dataType,
                         dst.dataType, dst.dataType, dst.wLayoutLog2,
                         dst.cLayoutLog2, dst.layoutOrder, &dst.tensor, &local_dst,
                         &zero, 0U);
    return commands;
}

static inline std::vector<NpuCmd>
primitiveReduceTemplate(uint32_t device_id, uint32_t op_code,
                        const PrimitiveVpu2DDesc &src,
                        const PrimitiveVpu2DDesc &dst,
                        const PrimitiveBufferAssignment &buffers)
{
    return primitiveUnaryTemplate(device_id, op_code, src, dst, buffers, 0U);
}

#endif
