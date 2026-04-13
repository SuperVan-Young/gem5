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

struct PrimitiveSyncDesc
{
    uint32_t syncIndicator = 0U;
    bool setSnsIndicator = false;
    bool launchSyncWait = false;
    uint32_t syncWaitWord1 = 0U;
    uint32_t syncWaitWord2 = 0U;
    uint32_t syncWaitWord3 = 0U;
};

enum
{
    PRIMITIVE_DLEN_BYTES = 8U,
    PRIMITIVE_LOCAL_BUFFER_STRIDE = 128U * 128U * sizeof(uint32_t),
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

static inline uint32_t
primitiveTensorSlotSpan(const PrimitiveTensorDesc &tensor)
{
    const uint64_t bytes = tensor.bytes();
    return static_cast<uint32_t>(
        (bytes + VPU_LOCAL_SLOT_STRIDE - 1U) / VPU_LOCAL_SLOT_STRIDE);
}

static inline VpuTensorDesc
primitiveLocalTensor(uint32_t base, uint32_t buffer_index,
                     const PrimitiveVpu2DDesc &desc)
{
    return vpu_tensor_desc(base + (buffer_index * PRIMITIVE_LOCAL_BUFFER_STRIDE),
                           desc.tensor.shape,
                           desc.tensor.stride);
}

static inline void
primitiveLaunchCommands(const std::vector<NpuCmd> &commands, uint64_t port_base)
{
    for (const auto &cmd : commands) {
        cmd.launchCmdAt(port_base);
    }
}

static inline void
primitiveSetSync(NpuCmd *cmd, const PrimitiveSyncDesc &sync)
{
    const uint32_t header = npuBuildHeaderWord(
        cmd->getDeviceType(),
        cmd->getDeviceId(),
        cmd->getOpCode(),
        sync.syncIndicator,
        sync.setSnsIndicator ? 1U : 0U,
        0U);
    cmd->setWord(0U, header);
}

static inline void
primitiveInitExecCmdFast(NpuCmd *cmd, uint32_t device_id, uint32_t op_code,
                         const PrimitiveVpu2DDesc &src0,
                         const PrimitiveVpu2DDesc &src1,
                         const PrimitiveVpu2DDesc &dst, uint32_t extra0)
{
    cmd->clear();
    cmd->setWord(0U, npuBuildHeaderWord(
                         NPU_DEVICE_TYPE_VPU, device_id, op_code, 0U, 0U, 0U));
    cmd->setWord(1U, vpu_pack_format_word(dst.dataType, src0.dataType,
                                          src1.dataType, dst.wLayoutLog2,
                                          dst.cLayoutLog2, dst.layoutOrder));
    cmd->setWord(VPU_CMD_WORD_DST_ADDR, dst.tensor.addr);
    cmd->setWord(VPU_CMD_WORD_DST_SHAPE, dst.tensor.shape);
    cmd->setWord(VPU_CMD_WORD_DST_STRIDE, dst.tensor.stride);
    cmd->setWord(VPU_CMD_WORD_SRC0_ADDR, src0.tensor.addr);
    cmd->setWord(VPU_CMD_WORD_SRC0_SHAPE, src0.tensor.shape);
    cmd->setWord(VPU_CMD_WORD_SRC0_STRIDE, src0.tensor.stride);
    cmd->setWord(VPU_CMD_WORD_SRC1_ADDR, src1.tensor.addr);
    cmd->setWord(VPU_CMD_WORD_SRC1_SHAPE, src1.tensor.shape);
    cmd->setWord(VPU_CMD_WORD_SRC1_STRIDE, src1.tensor.stride);
    cmd->setWord(VPU_CMD_WORD_EXTRA0, extra0);
}

static inline void
primitivePatchTensorAddr(NpuCmd *cmd, uint32_t addr_word_index, uint32_t addr)
{
    cmd->setWord(addr_word_index, addr);
}

static inline void
primitivePatchTensor(NpuCmd *cmd, uint32_t addr_word_index,
                     const VpuTensorDesc &desc)
{
    cmd->setWord(addr_word_index, desc.addr);
    cmd->setWord(addr_word_index + 1U, desc.shape);
    cmd->setWord(addr_word_index + 2U, desc.stride);
}

static inline void
primitiveMaybeLaunchSyncWait(uint32_t device_id, const PrimitiveSyncDesc &sync,
                             uint64_t port_base)
{
    if (!sync.launchSyncWait) {
        return;
    }

    if (sync.syncIndicator == 0U) {
        __builtin_trap();
    }

    npu_launch_sync_wait_at(device_id, sync.syncIndicator,
                            sync.syncWaitWord1, sync.syncWaitWord2,
                            sync.syncWaitWord3, port_base);
}

static inline NpuCmd
primitiveBuildUnaryExecCmd(uint32_t device_id, uint32_t op_code,
                           const PrimitiveVpu2DDesc &src,
                           const PrimitiveVpu2DDesc &dst, uint32_t extra0)
{
    const VpuTensorDesc zero = vpu_tensor_desc(0U, 0U, 0U);
    NpuCmd cmd;
    primitiveInitExecCmdFast(&cmd, device_id, op_code, src,
                             PrimitiveVpu2DDesc{zero, src.dataType}, dst,
                             extra0);
    return cmd;
}

static inline NpuCmd
primitiveBuildBinaryExecCmd(uint32_t device_id, uint32_t op_code,
                            const PrimitiveVpu2DDesc &src0,
                            const PrimitiveVpu2DDesc &src1,
                            const PrimitiveVpu2DDesc &dst)
{
    NpuCmd cmd;
    primitiveInitExecCmdFast(&cmd, device_id, op_code, src0, src1, dst, 0U);
    return cmd;
}

static inline NpuCmd
primitiveBuildReduceExecCmd(uint32_t device_id, uint32_t op_code,
                            const PrimitiveVpu2DDesc &src,
                            const PrimitiveVpu2DDesc &dst)
{
    const VpuTensorDesc zero = vpu_tensor_desc(0U, 0U, 0U);
    NpuCmd cmd;
    primitiveInitExecCmdFast(&cmd, device_id, op_code, src,
                             PrimitiveVpu2DDesc{zero, src.dataType}, dst, 0U);
    return cmd;
}

#endif
