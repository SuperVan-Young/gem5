#ifndef TESTS_GEM5_NPU_UTILS_NPU_SYNC_H_
#define TESTS_GEM5_NPU_UTILS_NPU_SYNC_H_

#include <stdint.h>

#include "cmd/common.hh"

#define NPU_SYNC_MMIO_BASE 0x71000000UL

enum NpuSyncOpcode {
    NPU_SYNC_OP_WAIT = 0x0U,
    NPU_SYNC_OP_SET = 0x1U,
};

static inline void
npuBuildSyncWaitCmd(NpuCmd *cmd, uint32_t device_id, uint32_t sync_indicator,
                    uint32_t w1, uint32_t w2, uint32_t w3)
{
    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_SYNC_INDICATOR_TABLE);
    cmd->setDeviceId(device_id);
    cmd->setOpCode(NPU_SYNC_OP_WAIT);
    cmd->setSyncIndicator(sync_indicator);
    cmd->clearCommonReservedBits();
    cmd->setWord(1U, w1);
    cmd->setWord(2U, w2);
    cmd->setWord(3U, w3);
}

static inline void
npu_launch_sync_wait_at(uint32_t device_id, uint32_t sync_indicator,
                        uint32_t w1, uint32_t w2, uint32_t w3,
                        uint64_t port_base)
{
    NpuCmd cmd;

    npuBuildSyncWaitCmd(&cmd, device_id, sync_indicator, w1, w2, w3);
    cmd.launchCmdAt(port_base);
}

static inline void
npu_launch_sync_wait(uint32_t device_id, uint32_t sync_indicator,
                     uint32_t w1, uint32_t w2, uint32_t w3)
{
    npu_launch_sync_wait_at(device_id, sync_indicator, w1, w2, w3,
                            NPU_CMD_PORT_BASE);
}

static inline uint32_t
npuBuildSyncSetWord(uint32_t device_id, uint32_t sync_indicator)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_SYNC_INDICATOR_TABLE);
    cmd.setDeviceId(device_id);
    cmd.setOpCode(NPU_SYNC_OP_SET);
    cmd.setSyncIndicator(sync_indicator);
    cmd.clearCommonReservedBits();
    return cmd.getWord(0U);
}

static inline void
npu_sync_signal_set(uint32_t device_id, uint32_t sync_indicator)
{
    const uint32_t cmd_word = npuBuildSyncSetWord(device_id, sync_indicator);

    npu_mmio_write32_one(NPU_SYNC_MMIO_BASE, cmd_word);
}

#endif
