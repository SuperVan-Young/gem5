#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_REDUCE_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_REDUCE_HH_

#include "internal.hh"

static inline size_t
vpu_primitive_reduce(uint32_t device_id, uint32_t op_code,
                     const PrimitiveTensorDesc &src,
                     const PrimitiveTensorDesc &dst,
                     const PrimitiveBufferAssignment &buffers = {},
                     const PrimitiveSyncDesc &sync = {},
                     uint64_t port_base = NPU_CMD_PORT_BASE)
{
    (void)buffers;
    const uint64_t slices = src.sliceCount(2U);
    if (dst.sliceCount(2U) != slices) {
        __builtin_trap();
    }

    for (uint64_t i = 0U; i < slices; ++i) {
        const PrimitiveVpu2DDesc slice_src =
            primitiveLowerLastAxis2D(primitivePeelTo2D(src, i));
        const PrimitiveVpu2DDesc slice_dst =
            primitiveLowerLastAxis2D(primitivePeelTo2D(dst, i));
        NpuCmd cmd = primitiveBuildReduceExecCmd(device_id, op_code, slice_src,
                                                 slice_dst);
        if (i + 1U == slices) {
            primitiveSetSync(&cmd, sync);
        }
        cmd.launchCmdAt(port_base);
    }

    primitiveMaybeLaunchSyncWait(device_id, sync, port_base);
    return slices;
}

#endif
