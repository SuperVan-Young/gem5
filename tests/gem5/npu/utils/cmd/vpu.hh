#ifndef TESTS_GEM5_NPU_UTILS_CMD_VPU_H_
#define TESTS_GEM5_NPU_UTILS_CMD_VPU_H_

#include <stdint.h>

#include "../npu_sync.hh"
#include "common.hh"

enum VpuOpcode
{
    VPU_OP_EXEC = 0x0U,
    VPU_OP_VADD = 0x1U,
    VPU_OP_VSUB = 0x2U,
    VPU_OP_VMUL = 0x3U,
    VPU_OP_VDIV = 0x4U,
    VPU_OP_VSCALE = 0x5U,
    VPU_OP_VCVT_I2F = 0x6U,
    VPU_OP_VCVT_F2I = 0x7U,
    VPU_OP_VSQRT = 0x8U,
    VPU_OP_VFMA = 0x9U,
    VPU_OP_VREDUCE_SUM = 0xAU,
    VPU_OP_VREDUCE_MAX = 0xBU,
    VPU_OP_VLOAD = 0xCU,
    VPU_OP_VSTORE = 0xDU,
    VPU_OP_VEXP = 0xEU,
    VPU_OP_VSOFTMAX = 0xFU,
};

enum VpuDataType
{
    VPU_DATA_I32 = 0x0U,
    VPU_DATA_F32 = 0x1U,
};

enum VpuCmdWord
{
    VPU_CMD_WORD_READ_MASK = 1U,
    VPU_CMD_WORD_WRITE_MASK = 2U,
    VPU_CMD_WORD_REPETITION = 3U,
    VPU_CMD_WORD_FLAGS = 4U,
    VPU_CMD_WORD_ELEM_COUNT = 5U,
    VPU_CMD_WORD_SRC_STRIDE = 6U,
    VPU_CMD_WORD_DST_STRIDE = 7U,
    VPU_CMD_WORD_DATA_TYPE = 8U,
    VPU_CMD_WORD_SCALAR_BITS = 9U,
    VPU_CMD_WORD_SRC0_ADDR = 10U,
    VPU_CMD_WORD_SRC1_ADDR = 11U,
    VPU_CMD_WORD_SRC2_ADDR = 12U,
    VPU_CMD_WORD_DST_ADDR = 13U,
};

struct VpuCmdInstr
{
    uint32_t header;
    uint32_t read_mask;
    uint32_t write_mask;
    uint32_t repetition;
    uint32_t flags;
    uint32_t elem_count;
    uint32_t src_stride_bytes;
    uint32_t dst_stride_bytes;
    uint32_t data_type;
    uint32_t scalar_bits;
    uint32_t src0_addr;
    uint32_t src1_addr;
    uint32_t src2_addr;
    uint32_t dst_addr;
    uint32_t reserved0;
    uint32_t reserved1;
};

static_assert(sizeof(VpuCmdInstr) == NPU_CMD_BUFFER_BYTES,
              "VPU command struct must remain 64 bytes.");

enum VpuAddressLayout
{
    VPU_LOCAL_INPUT_BASE = 0x80000000U,
    VPU_LOCAL_OUTPUT_BASE = 0x81000000U,
    VPU_LOCAL_SLOT_STRIDE = 0x40U,
    VPU_DEFAULT_INPUT_BUFFER = 0U,
    VPU_DEFAULT_OUTPUT_BUFFER = 0U,
};

static inline uint32_t
vpu_first_port(uint32_t mask)
{
    uint32_t port = 0U;
    while (((mask >> port) & 0x1U) == 0U) {
        ++port;
    }
    return port;
}

static inline uint32_t
vpu_local_addr(uint32_t base, uint32_t buffer_index)
{
    return base + (buffer_index * VPU_LOCAL_SLOT_STRIDE);
}

static inline void
vpu_cmd_init_raw(NpuCmd *cmd, uint32_t device_id, uint32_t op_code,
                 uint32_t sync_indicator)
{
    VpuCmdInstr vpu_cmd = {};
    NpuCmdBinaryData binary = {};

    vpu_cmd.header = npuBuildHeaderWord(
        NPU_DEVICE_TYPE_VPU, device_id, op_code, sync_indicator,
        sync_indicator != 0U ? 1U : 0U, 0U);
    npuBinaryDataFromObject(&binary, vpu_cmd);
    cmd->loadBinary(binary);
}

static inline void
vpu_cmd_set_common_fields(NpuCmd *cmd, uint32_t read_mask,
                          uint32_t write_mask, uint32_t repetition,
                          uint32_t flags, uint32_t elem_count,
                          uint32_t src_stride_bytes,
                          uint32_t dst_stride_bytes, uint32_t data_type,
                          uint32_t scalar_bits, uint32_t src0_addr,
                          uint32_t src1_addr, uint32_t src2_addr,
                          uint32_t dst_addr)
{
    NpuCmdBinaryData binary = {};
    VpuCmdInstr vpu_cmd = {};

    cmd->copyBinaryData(&binary);
    npuObjectFromBinaryData(&vpu_cmd, binary);
    vpu_cmd.read_mask = read_mask;
    vpu_cmd.write_mask = write_mask;
    vpu_cmd.repetition = repetition;
    vpu_cmd.flags = flags;
    vpu_cmd.elem_count = elem_count;
    vpu_cmd.src_stride_bytes = src_stride_bytes;
    vpu_cmd.dst_stride_bytes = dst_stride_bytes;
    vpu_cmd.data_type = data_type;
    vpu_cmd.scalar_bits = scalar_bits;
    vpu_cmd.src0_addr = src0_addr;
    vpu_cmd.src1_addr = src1_addr;
    vpu_cmd.src2_addr = src2_addr;
    vpu_cmd.dst_addr = dst_addr;
    npuBinaryDataFromObject(&binary, vpu_cmd);
    cmd->loadBinary(binary);
}

static inline void
vpu_cmd_launch_load_one(uint32_t device_id, uint32_t sync_indicator,
                        uint32_t port, uint32_t elem_count,
                        uint32_t src_stride_bytes, uint32_t data_type,
                        uint32_t input_buffer_index)
{
    NpuCmd cmd;
    vpu_cmd_init_raw(&cmd, device_id, VPU_OP_VLOAD, sync_indicator);
    vpu_cmd_set_common_fields(
        &cmd, 1U << port, 0U, 1U, 0U, elem_count, src_stride_bytes, 0U,
        data_type, 0U, 0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE), 0U, 0U,
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, input_buffer_index));
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_store_one(uint32_t device_id, uint32_t sync_indicator,
                         uint32_t port, uint32_t elem_count,
                         uint32_t dst_stride_bytes, uint32_t data_type,
                         uint32_t source_local_base,
                         uint32_t buffer_index)
{
    NpuCmd cmd;
    vpu_cmd_init_raw(&cmd, device_id, VPU_OP_VSTORE, sync_indicator);
    vpu_cmd_set_common_fields(
        &cmd, 0U, 1U << port, 1U, 0U, elem_count, 0U, dst_stride_bytes,
        data_type, 0U,
        vpu_local_addr(source_local_base, buffer_index), 0U, 0U,
        0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE));
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_compute(uint32_t device_id, uint32_t op_code,
                       uint32_t sync_indicator, uint32_t read_mask,
                       uint32_t write_mask, uint32_t repetition,
                       uint32_t elem_count, uint32_t src_stride_bytes,
                       uint32_t dst_stride_bytes, uint32_t data_type,
                       uint32_t scalar_bits)
{
    NpuCmd cmd;
    vpu_cmd_init_raw(&cmd, device_id, op_code, sync_indicator);
    vpu_cmd_set_common_fields(
        &cmd, read_mask, write_mask, repetition, 0U, elem_count,
        src_stride_bytes, dst_stride_bytes, data_type, scalar_bits,
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER));
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_binary(uint32_t device_id, uint32_t op_code,
                      uint32_t sync_indicator, uint32_t read_mask,
                      uint32_t write_mask, uint32_t repetition,
                      uint32_t elem_count, uint32_t src_stride_bytes,
                      uint32_t dst_stride_bytes, uint32_t data_type)
{
    uint32_t remaining = read_mask;
    while (remaining != 0U) {
        const uint32_t port = vpu_first_port(remaining);
        remaining &= ~(1U << port);
        vpu_cmd_launch_load_one(device_id, 0U, port, elem_count,
                                src_stride_bytes, data_type,
                                VPU_DEFAULT_INPUT_BUFFER);
    }

    vpu_cmd_launch_compute(device_id, op_code, 0U, read_mask,
                           write_mask, repetition, elem_count,
                           src_stride_bytes, dst_stride_bytes, data_type, 0U);

    uint32_t remaining_writes = write_mask;
    while (remaining_writes != 0U) {
        const uint32_t port = vpu_first_port(remaining_writes);
        remaining_writes &= ~(1U << port);
        const uint32_t store_sync =
            remaining_writes == 0U ? sync_indicator : 0U;
        vpu_cmd_launch_store_one(device_id, store_sync, port, elem_count,
                                 dst_stride_bytes, data_type,
                                 VPU_LOCAL_OUTPUT_BASE,
                                 VPU_DEFAULT_OUTPUT_BUFFER);
    }
}

static inline void
vpu_cmd_launch_binary_at(uint64_t port_base, uint32_t device_id,
                         uint32_t op_code, uint32_t sync_indicator,
                         uint32_t read_mask, uint32_t write_mask,
                         uint32_t repetition, uint32_t elem_count,
                         uint32_t src_stride_bytes,
                         uint32_t dst_stride_bytes, uint32_t data_type)
{
    (void)port_base;
    vpu_cmd_launch_binary(device_id, op_code, sync_indicator, read_mask,
                          write_mask, repetition, elem_count,
                          src_stride_bytes, dst_stride_bytes, data_type);
}

static inline void
vpu_cmd_launch_unary(uint32_t device_id, uint32_t op_code,
                     uint32_t sync_indicator, uint32_t read_mask,
                     uint32_t write_mask, uint32_t repetition,
                     uint32_t elem_count, uint32_t src_stride_bytes,
                     uint32_t dst_stride_bytes, uint32_t data_type)
{
    const uint32_t store_elem_count =
        (op_code == VPU_OP_VREDUCE_SUM || op_code == VPU_OP_VREDUCE_MAX) ?
        1U : elem_count;

    uint32_t remaining = read_mask;
    while (remaining != 0U) {
        const uint32_t port = vpu_first_port(remaining);
        remaining &= ~(1U << port);
        vpu_cmd_launch_load_one(device_id, 0U, port, elem_count,
                                src_stride_bytes, data_type,
                                VPU_DEFAULT_INPUT_BUFFER);
    }

    vpu_cmd_launch_compute(device_id, op_code, 0U, read_mask,
                           write_mask, repetition, elem_count,
                           src_stride_bytes, dst_stride_bytes, data_type, 0U);

    uint32_t remaining_writes = write_mask;
    while (remaining_writes != 0U) {
        const uint32_t port = vpu_first_port(remaining_writes);
        remaining_writes &= ~(1U << port);
        const uint32_t store_sync =
            remaining_writes == 0U ? sync_indicator : 0U;
        vpu_cmd_launch_store_one(device_id, store_sync, port, store_elem_count,
                                 dst_stride_bytes, data_type,
                                 VPU_LOCAL_OUTPUT_BASE,
                                 VPU_DEFAULT_OUTPUT_BUFFER);
    }
}

static inline void
vpu_cmd_launch_unary_at(uint64_t port_base, uint32_t device_id,
                        uint32_t op_code, uint32_t sync_indicator,
                        uint32_t read_mask, uint32_t write_mask,
                        uint32_t repetition, uint32_t elem_count,
                        uint32_t src_stride_bytes,
                        uint32_t dst_stride_bytes, uint32_t data_type)
{
    (void)port_base;
    vpu_cmd_launch_unary(device_id, op_code, sync_indicator, read_mask,
                         write_mask, repetition, elem_count,
                         src_stride_bytes, dst_stride_bytes, data_type);
}

static inline void
vpu_cmd_launch_ternary(uint32_t device_id, uint32_t op_code,
                       uint32_t sync_indicator, uint32_t read_mask,
                       uint32_t write_mask, uint32_t repetition,
                       uint32_t elem_count, uint32_t src_stride_bytes,
                       uint32_t dst_stride_bytes, uint32_t data_type)
{
    uint32_t remaining = read_mask;
    while (remaining != 0U) {
        const uint32_t port = vpu_first_port(remaining);
        remaining &= ~(1U << port);
        vpu_cmd_launch_load_one(device_id, 0U, port, elem_count,
                                src_stride_bytes, data_type,
                                VPU_DEFAULT_INPUT_BUFFER);
    }

    vpu_cmd_launch_compute(device_id, op_code, 0U, read_mask,
                           write_mask, repetition, elem_count,
                           src_stride_bytes, dst_stride_bytes, data_type, 0U);

    uint32_t remaining_writes = write_mask;
    while (remaining_writes != 0U) {
        const uint32_t port = vpu_first_port(remaining_writes);
        remaining_writes &= ~(1U << port);
        const uint32_t store_sync =
            remaining_writes == 0U ? sync_indicator : 0U;
        vpu_cmd_launch_store_one(device_id, store_sync, port, elem_count,
                                 dst_stride_bytes, data_type,
                                 VPU_LOCAL_OUTPUT_BASE,
                                 VPU_DEFAULT_OUTPUT_BUFFER);
    }
}

static inline void
vpu_cmd_launch_ternary_at(uint64_t port_base, uint32_t device_id,
                          uint32_t op_code, uint32_t sync_indicator,
                          uint32_t read_mask, uint32_t write_mask,
                          uint32_t repetition, uint32_t elem_count,
                          uint32_t src_stride_bytes,
                          uint32_t dst_stride_bytes, uint32_t data_type)
{
    (void)port_base;
    vpu_cmd_launch_ternary(device_id, op_code, sync_indicator, read_mask,
                           write_mask, repetition, elem_count,
                           src_stride_bytes, dst_stride_bytes, data_type);
}

static inline void
vpu_cmd_launch_scale(uint32_t device_id, uint32_t sync_indicator,
                     uint32_t read_mask, uint32_t write_mask,
                     uint32_t repetition, uint32_t elem_count,
                     uint32_t src_stride_bytes, uint32_t dst_stride_bytes,
                     uint32_t data_type, uint32_t scalar_bits)
{
    uint32_t remaining = read_mask;
    while (remaining != 0U) {
        const uint32_t port = vpu_first_port(remaining);
        remaining &= ~(1U << port);
        vpu_cmd_launch_load_one(device_id, 0U, port, elem_count,
                                src_stride_bytes, data_type,
                                VPU_DEFAULT_INPUT_BUFFER);
    }

    vpu_cmd_launch_compute(device_id, VPU_OP_VSCALE, 0U, read_mask,
                           write_mask, repetition, elem_count,
                           src_stride_bytes, dst_stride_bytes, data_type,
                           scalar_bits);

    uint32_t remaining_writes = write_mask;
    while (remaining_writes != 0U) {
        const uint32_t port = vpu_first_port(remaining_writes);
        remaining_writes &= ~(1U << port);
        const uint32_t store_sync =
            remaining_writes == 0U ? sync_indicator : 0U;
        vpu_cmd_launch_store_one(device_id, store_sync, port, elem_count,
                                 dst_stride_bytes, data_type,
                                 VPU_LOCAL_OUTPUT_BASE,
                                 VPU_DEFAULT_OUTPUT_BUFFER);
    }
}

static inline void
vpu_cmd_launch_scale_at(uint64_t port_base, uint32_t device_id,
                        uint32_t sync_indicator, uint32_t read_mask,
                        uint32_t write_mask, uint32_t repetition,
                        uint32_t elem_count, uint32_t src_stride_bytes,
                        uint32_t dst_stride_bytes, uint32_t data_type,
                        uint32_t scalar_bits)
{
    (void)port_base;
    vpu_cmd_launch_scale(device_id, sync_indicator, read_mask, write_mask,
                         repetition, elem_count, src_stride_bytes,
                         dst_stride_bytes, data_type, scalar_bits);
}

static inline void
vpu_cmd_launch_load(uint32_t device_id, uint32_t sync_indicator,
                    uint32_t read_mask, uint32_t elem_count,
                    uint32_t src_stride_bytes, uint32_t data_type)
{
    uint32_t remaining = read_mask;
    while (remaining != 0U) {
        const uint32_t port = vpu_first_port(remaining);
        remaining &= ~(1U << port);
        const uint32_t load_sync = remaining == 0U ? sync_indicator : 0U;
        vpu_cmd_launch_load_one(device_id, load_sync, port, elem_count,
                                src_stride_bytes, data_type,
                                VPU_DEFAULT_INPUT_BUFFER);
    }
}

static inline void
vpu_cmd_launch_store(uint32_t device_id, uint32_t sync_indicator,
                     uint32_t write_mask, uint32_t elem_count,
                     uint32_t dst_stride_bytes, uint32_t data_type)
{
    uint32_t remaining = write_mask;
    while (remaining != 0U) {
        const uint32_t port = vpu_first_port(remaining);
        remaining &= ~(1U << port);
        const uint32_t store_sync = remaining == 0U ? sync_indicator : 0U;
        vpu_cmd_launch_store_one(device_id, store_sync, port, elem_count,
                                 dst_stride_bytes, data_type,
                                 VPU_LOCAL_INPUT_BASE,
                                 VPU_DEFAULT_INPUT_BUFFER);
    }
}

static inline void
vpu_cmd_launch_legacy_exec(uint32_t device_id, uint32_t sync_indicator,
                           uint32_t read_mask, uint32_t write_mask,
                           uint32_t repetition)
{
    for (uint32_t iteration = 0U; iteration < repetition; ++iteration) {
        uint32_t load_mask = read_mask | write_mask;

        while (load_mask != 0U) {
            const uint32_t port = vpu_first_port(load_mask);
            load_mask &= ~(1U << port);
            vpu_cmd_launch_load_one(device_id, 0U, port, 1U,
                                    sizeof(uint32_t), VPU_DATA_I32,
                                    VPU_DEFAULT_INPUT_BUFFER);
        }

        vpu_cmd_launch_compute(device_id, VPU_OP_EXEC, 0U, read_mask,
                               write_mask, 1U, 1U, sizeof(uint32_t),
                               sizeof(uint32_t), VPU_DATA_I32, 0U);

        uint32_t remaining_writes = write_mask;
        while (remaining_writes != 0U) {
            const uint32_t port = vpu_first_port(remaining_writes);
            remaining_writes &= ~(1U << port);
            const uint32_t store_sync =
                (iteration + 1U == repetition && remaining_writes == 0U) ?
                sync_indicator : 0U;
            vpu_cmd_launch_store_one(device_id, store_sync, port, 1U,
                                     sizeof(uint32_t), VPU_DATA_I32,
                                     VPU_LOCAL_OUTPUT_BASE,
                                     VPU_DEFAULT_OUTPUT_BUFFER);
        }
    }

    if (sync_indicator != 0U) {
        npu_launch_sync_wait(device_id, sync_indicator, 0U, 0U, 0U);
        npu_cmd_sync_done();
    }
}

#endif
