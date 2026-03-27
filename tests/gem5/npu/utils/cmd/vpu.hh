#ifndef TESTS_GEM5_NPU_UTILS_CMD_VPU_H_
#define TESTS_GEM5_NPU_UTILS_CMD_VPU_H_

#include <stdint.h>

#include "common.hh"

enum VpuOpcode {
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

enum VpuDataType {
    VPU_DATA_I32 = 0x0U,
    VPU_DATA_F32 = 0x1U,
};

enum VpuCmdWord {
    VPU_CMD_WORD_READ_MASK = 1U,
    VPU_CMD_WORD_WRITE_MASK = 2U,
    VPU_CMD_WORD_REPETITION = 3U,
    VPU_CMD_WORD_FLAGS = 4U,
    VPU_CMD_WORD_ELEM_COUNT = 5U,
    VPU_CMD_WORD_SRC_STRIDE = 6U,
    VPU_CMD_WORD_DST_STRIDE = 7U,
    VPU_CMD_WORD_DATA_TYPE = 8U,
    VPU_CMD_WORD_SCALAR_BITS = 9U,
};

static inline void
vpu_cmd_init(NpuCmd *cmd, uint32_t device_id, uint32_t op_code,
             uint32_t sync_indicator, uint32_t read_mask,
             uint32_t write_mask, uint32_t repetition)
{
    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd->setDeviceId(device_id);
    cmd->setOpCode(op_code);
    cmd->setSyncIndicator(sync_indicator);
    if (sync_indicator != 0U) {
        cmd->setSetIndicatorSns(1U);
    }
    cmd->clearCommonReservedBits();
    cmd->setWord(VPU_CMD_WORD_READ_MASK, read_mask);
    cmd->setWord(VPU_CMD_WORD_WRITE_MASK, write_mask);
    cmd->setWord(VPU_CMD_WORD_REPETITION, repetition);
}

static inline void
vpu_cmd_set_vector_fields(NpuCmd *cmd, uint32_t flags, uint32_t elem_count,
                          uint32_t src_stride_bytes,
                          uint32_t dst_stride_bytes, uint32_t data_type)
{
    cmd->setWord(VPU_CMD_WORD_FLAGS, flags);
    cmd->setWord(VPU_CMD_WORD_ELEM_COUNT, elem_count);
    cmd->setWord(VPU_CMD_WORD_SRC_STRIDE, src_stride_bytes);
    cmd->setWord(VPU_CMD_WORD_DST_STRIDE, dst_stride_bytes);
    cmd->setWord(VPU_CMD_WORD_DATA_TYPE, data_type);
}

static inline void
vpu_cmd_set_scalar_bits(NpuCmd *cmd, uint32_t scalar_bits)
{
    cmd->setWord(VPU_CMD_WORD_SCALAR_BITS, scalar_bits);
}

static inline void
vpu_cmd_launch_binary(uint32_t device_id, uint32_t op_code,
                      uint32_t sync_indicator, uint32_t read_mask,
                      uint32_t write_mask, uint32_t repetition,
                      uint32_t elem_count, uint32_t src_stride_bytes,
                      uint32_t dst_stride_bytes, uint32_t data_type)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, op_code, sync_indicator, read_mask,
                 write_mask, repetition);
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes,
                              dst_stride_bytes, data_type);
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_binary_at(uint64_t port_base, uint32_t device_id,
                         uint32_t op_code, uint32_t sync_indicator,
                         uint32_t read_mask, uint32_t write_mask,
                         uint32_t repetition, uint32_t elem_count,
                         uint32_t src_stride_bytes,
                         uint32_t dst_stride_bytes, uint32_t data_type)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, op_code, sync_indicator, read_mask,
                 write_mask, repetition);
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes,
                              dst_stride_bytes, data_type);
    cmd.launchCmdAt(port_base);
}

static inline void
vpu_cmd_launch_unary(uint32_t device_id, uint32_t op_code,
                     uint32_t sync_indicator, uint32_t read_mask,
                     uint32_t write_mask, uint32_t repetition,
                     uint32_t elem_count, uint32_t src_stride_bytes,
                     uint32_t dst_stride_bytes, uint32_t data_type)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, op_code, sync_indicator, read_mask,
                 write_mask, repetition);
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes,
                              dst_stride_bytes, data_type);
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_unary_at(uint64_t port_base, uint32_t device_id,
                        uint32_t op_code, uint32_t sync_indicator,
                        uint32_t read_mask, uint32_t write_mask,
                        uint32_t repetition, uint32_t elem_count,
                        uint32_t src_stride_bytes,
                        uint32_t dst_stride_bytes, uint32_t data_type)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, op_code, sync_indicator, read_mask,
                 write_mask, repetition);
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes,
                              dst_stride_bytes, data_type);
    cmd.launchCmdAt(port_base);
}

static inline void
vpu_cmd_launch_ternary(uint32_t device_id, uint32_t op_code,
                       uint32_t sync_indicator, uint32_t read_mask,
                       uint32_t write_mask, uint32_t repetition,
                       uint32_t elem_count, uint32_t src_stride_bytes,
                       uint32_t dst_stride_bytes, uint32_t data_type)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, op_code, sync_indicator, read_mask,
                 write_mask, repetition);
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes,
                              dst_stride_bytes, data_type);
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_ternary_at(uint64_t port_base, uint32_t device_id,
                          uint32_t op_code, uint32_t sync_indicator,
                          uint32_t read_mask, uint32_t write_mask,
                          uint32_t repetition, uint32_t elem_count,
                          uint32_t src_stride_bytes,
                          uint32_t dst_stride_bytes, uint32_t data_type)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, op_code, sync_indicator, read_mask,
                 write_mask, repetition);
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes,
                              dst_stride_bytes, data_type);
    cmd.launchCmdAt(port_base);
}

static inline void
vpu_cmd_launch_scale(uint32_t device_id, uint32_t sync_indicator,
                     uint32_t read_mask, uint32_t write_mask,
                     uint32_t repetition, uint32_t elem_count,
                     uint32_t src_stride_bytes, uint32_t dst_stride_bytes,
                     uint32_t data_type, uint32_t scalar_bits)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, VPU_OP_VSCALE, sync_indicator, read_mask,
                 write_mask, repetition);
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes,
                              dst_stride_bytes, data_type);
    vpu_cmd_set_scalar_bits(&cmd, scalar_bits);
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_scale_at(uint64_t port_base, uint32_t device_id,
                        uint32_t sync_indicator, uint32_t read_mask,
                        uint32_t write_mask, uint32_t repetition,
                        uint32_t elem_count, uint32_t src_stride_bytes,
                        uint32_t dst_stride_bytes, uint32_t data_type,
                        uint32_t scalar_bits)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, VPU_OP_VSCALE, sync_indicator, read_mask,
                 write_mask, repetition);
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes,
                              dst_stride_bytes, data_type);
    vpu_cmd_set_scalar_bits(&cmd, scalar_bits);
    cmd.launchCmdAt(port_base);
}

static inline void
vpu_cmd_launch_load(uint32_t device_id, uint32_t sync_indicator,
                    uint32_t read_mask, uint32_t elem_count,
                    uint32_t src_stride_bytes, uint32_t data_type)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, VPU_OP_VLOAD, sync_indicator, read_mask,
                 0U, 1U);
    if (sync_indicator != 0U) {
        cmd.setSetIndicatorSns(1U);
    }
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, src_stride_bytes, 0U,
                              data_type);
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_store(uint32_t device_id, uint32_t sync_indicator,
                     uint32_t write_mask, uint32_t elem_count,
                     uint32_t dst_stride_bytes, uint32_t data_type)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, VPU_OP_VSTORE, sync_indicator, 0U,
                 write_mask, 1U);
    if (sync_indicator != 0U) {
        cmd.setSetIndicatorSns(1U);
    }
    vpu_cmd_set_vector_fields(&cmd, 0U, elem_count, 0U, dst_stride_bytes,
                              data_type);
    cmd.launchCmd();
}

static inline void
vpu_cmd_launch_legacy_exec(uint32_t device_id, uint32_t sync_indicator,
                           uint32_t read_mask, uint32_t write_mask,
                           uint32_t repetition)
{
    NpuCmd cmd;
    vpu_cmd_init(&cmd, device_id, VPU_OP_EXEC, sync_indicator, read_mask,
                 write_mask, repetition);
    cmd.setWord(VPU_CMD_WORD_FLAGS, 0U);
    cmd.launchCmd();
}

#endif
