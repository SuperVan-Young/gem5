/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>

#include "cmd/common.hh"

enum VpuOpcode
{
    VPU_OP_EXEC = 0x0U,
};

#define DEVICE_ID 0x0U
#define NUM_CMDS 5
#define SEU_CMD_READ_MASK 0x00000000U
#define SEU_CMD_WRITE_MASK 0x00000000U
#define SEU_CMD_REPETITION 0x00000001U
#define SEU_CMD_RESERVED 0x00000000U

static void
launch_seu_cmd(uint32_t cmd_id)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(DEVICE_ID);
    cmd.setOpCode(VPU_OP_EXEC);
    cmd.setSyncIndicator(cmd_id);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, SEU_CMD_READ_MASK);
    cmd.setWord(2U, SEU_CMD_WRITE_MASK);
    cmd.setWord(3U, SEU_CMD_REPETITION);
    cmd.setWord(4U, SEU_CMD_RESERVED);
    cmd.launchCmd();
}

int
main(void)
{
    for (uint32_t i = 0; i < NUM_CMDS; ++i) {
        launch_seu_cmd(i);
    }

    return 0;
}
