#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_BINARY_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_BINARY_HH_

#include "internal.hh"

static inline size_t
vpu_primitive_binary(uint32_t device_id, uint32_t op_code,
                     const PrimitiveTensorDesc &src0,
                     const PrimitiveTensorDesc &src1,
                     const PrimitiveTensorDesc &dst,
                     const PrimitiveBufferAssignment &buffers = {},
                     const PrimitiveSyncDesc &sync = {},
                     uint64_t port_base = NPU_CMD_PORT_BASE)
{
    (void)buffers;
    const uint64_t slices = dst.sliceCount(2U);
    const uint64_t src0_slices = src0.sliceCount(2U);
    const uint64_t src1_slices = src1.sliceCount(2U);
    if ((src0_slices != 1U && src0_slices != slices) ||
        (src1_slices != 1U && src1_slices != slices)) {
        __builtin_trap();
    }

    for (uint64_t i = 0U; i < slices; ++i) {
        const PrimitiveVpu2DDesc slice_src0 = primitiveLowerLastAxis2D(
            primitivePeelTo2D(src0, src0_slices == 1U ? 0U : i));
        const PrimitiveVpu2DDesc slice_src1 = primitiveLowerLastAxis2D(
            primitivePeelTo2D(src1, src1_slices == 1U ? 0U : i));
        const PrimitiveVpu2DDesc slice_dst =
            primitiveLowerLastAxis2D(primitivePeelTo2D(dst, i));
        NpuCmd cmd = primitiveBuildBinaryExecCmd(device_id, op_code, slice_src0,
                                                 slice_src1, slice_dst);
        if (i + 1U == slices) {
            primitiveSetSync(&cmd, sync);
        }
        cmd.launchCmdAt(port_base);
    }

    primitiveMaybeLaunchSyncWait(device_id, sync, port_base);
    return slices;
}

#endif
