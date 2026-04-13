#ifndef TESTS_GEM5_NPU_UTILS_CMD_MPU_H_
#define TESTS_GEM5_NPU_UTILS_CMD_MPU_H_

#include <stdint.h>

#include "common.hh"

#define MPU_CMD_READ_MASK 0x00000000U
#define MPU_CMD_WRITE_MASK 0x00000000U
#define MPU_CMD_REPETITION 0x00000001U
#define MPU_CMD_RESERVED 0x00000000U

#define MPU_WORD_BUFFER 5U
#define MPU_WORD_M 6U
#define MPU_WORD_N 7U
#define MPU_WORD_K 8U
#define MPU_WORD_SPM_ADDR_LO 9U
#define MPU_WORD_SPM_ADDR_HI 10U
#define MPU_WORD_STRIDE 11U
#define MPU_WORD_FLAGS 12U

#define MPU_BUFFER_A 0x0U
#define MPU_BUFFER_B 0x1U
#define MPU_BUFFER_C 0x2U
#define MPU_BUFFER_RESERVED 0x3U

#define MPU_DATA_TYPE_INT8 0x0U
#define MPU_OP_MVIN 0x0U
#define MPU_OP_MVOUT 0x1U
#define MPU_OP_LOAD 0x2U
#define MPU_OP_COMPUTE 0x3U
#define MPU_OP_DRAIN 0x4U

static inline uint32_t
mpu_op_code(uint32_t data_type, uint32_t cmd_kind)
{
    return ((data_type & 0x7U) << 5) | (cmd_kind & 0x1fU);
}

static inline uint32_t
mpu_buffer_word(uint32_t buffer_kind, uint32_t buffer_index)
{
    return (buffer_kind & 0x3U) | ((buffer_index & 0x1U) << 2);
}

static inline void
mpu_init_common_words(NpuCmd *cmd)
{
    cmd->setWord(1U, MPU_CMD_READ_MASK);
    cmd->setWord(2U, MPU_CMD_WRITE_MASK);
    cmd->setWord(3U, MPU_CMD_REPETITION);
    cmd->setWord(4U, MPU_CMD_RESERVED);
    cmd->setWord(MPU_WORD_FLAGS, 0U);
    cmd->setWord(13U, 0U);
    cmd->setWord(14U, 0U);
    cmd->setWord(15U, 0U);
}

static inline void
mpu_build_cmd(NpuCmd *cmd, uint32_t device_id, uint32_t op_code,
              uint32_t buffer_kind, uint32_t buffer_index,
              uint32_t m, uint32_t n, uint32_t k,
              uint64_t spm_addr, uint32_t stride_bytes,
              uint32_t sync_indicator, uint32_t set_completion_sync)
{
    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_MPU);
    cmd->setDeviceId(device_id);
    cmd->setOpCode(op_code);
    cmd->setSyncIndicator(sync_indicator);
    cmd->setSetIndicatorSns(set_completion_sync ? 1U : 0U);
    cmd->setSetIndicatorSnd(0U);
    cmd->clearCommonReservedBits();
    mpu_init_common_words(cmd);
    cmd->setWord(MPU_WORD_BUFFER, mpu_buffer_word(buffer_kind, buffer_index));
    cmd->setWord(MPU_WORD_M, m);
    cmd->setWord(MPU_WORD_N, n);
    cmd->setWord(MPU_WORD_K, k);
    cmd->setWord(MPU_WORD_SPM_ADDR_LO, (uint32_t)(spm_addr & 0xffffffffULL));
    cmd->setWord(MPU_WORD_SPM_ADDR_HI, (uint32_t)(spm_addr >> 32));
    cmd->setWord(MPU_WORD_STRIDE, stride_bytes);
}

static inline void
mpu_launch_cmd(uint32_t device_id, uint32_t op_code,
               uint32_t buffer_kind, uint32_t buffer_index,
               uint32_t m, uint32_t n, uint32_t k,
               uint64_t spm_addr, uint32_t stride_bytes,
               uint32_t sync_indicator, uint32_t set_completion_sync)
{
    NpuCmd cmd;

    mpu_build_cmd(&cmd, device_id, op_code, buffer_kind, buffer_index,
                  m, n, k, spm_addr, stride_bytes,
                  sync_indicator, set_completion_sync);
    cmd.launchCmd();
}

#endif
