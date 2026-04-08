#ifndef TESTS_GEM5_NPU_UTILS_NPU_SYNC_H_
#define TESTS_GEM5_NPU_UTILS_NPU_SYNC_H_

#include <stdint.h>

#include "cmd/common.hh"

#define NPU_SYNC_MMIO_BASE 0x71000000UL
#define NPU_CMD_CTRL_SYNC_DONE 2U

enum NpuSyncOpcode {
    NPU_SYNC_OP_WAIT = 0x0U,
    NPU_SYNC_OP_SET = 0x1U,
};

struct SyncCmdInstr
{
    uint32_t header;
    uint32_t word1;
    uint32_t word2;
    uint32_t word3;
    uint32_t reserved[12];
};

static_assert(sizeof(SyncCmdInstr) == NPU_CMD_BUFFER_BYTES,
              "Sync command struct must remain 64 bytes.");

static inline void
npu_cmd_sync_done_at(uint64_t port_base)
{
    npu_mmio_write32_one(NPU_CMD_CTRL_ADDR(port_base), NPU_CMD_CTRL_SYNC_DONE);
}

static inline void
npu_cmd_sync_done(void)
{
    npu_cmd_sync_done_at(NPU_CMD_PORT_BASE);
}

static inline void
npuBuildSyncWaitCmd(NpuCmd *cmd, uint32_t device_id, uint32_t sync_indicator,
                    uint32_t w1, uint32_t w2, uint32_t w3)
{
    SyncCmdInstr sync_cmd = {};
    NpuCmdBinaryData binary = {};

    sync_cmd.header = npuBuildHeaderWord(
        NPU_DEVICE_TYPE_SYNC_INDICATOR_TABLE, device_id, NPU_SYNC_OP_WAIT,
        sync_indicator, 0U, 0U);
    sync_cmd.word1 = w1;
    sync_cmd.word2 = w2;
    sync_cmd.word3 = w3;
    npuBinaryDataFromObject(&binary, sync_cmd);
    cmd->loadBinary(binary);
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
    return npuBuildHeaderWord(NPU_DEVICE_TYPE_SYNC_INDICATOR_TABLE, device_id,
                              NPU_SYNC_OP_SET, sync_indicator, 0U, 0U);
}

static inline void
npu_sync_signal_set(uint32_t device_id, uint32_t sync_indicator)
{
    const uint32_t cmd_word = npuBuildSyncSetWord(device_id, sync_indicator);

    npu_mmio_write32_one(NPU_SYNC_MMIO_BASE, cmd_word);
}

#endif
