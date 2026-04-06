/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>

#include "cmd/common.hh"
#include "npu_sync.hh"

enum SeuOpcode
{
    SEU_OP_EXEC = 0x0U,
    SEU_OP_LOAD = 0x1U,
    SEU_OP_STORE = 0x2U,
};

static void
launch_cmd(uint32_t opcode, uint32_t queue_select, uint32_t uop_count,
           uint32_t aux0, uint32_t aux1, uint32_t sync_id)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(0U);
    cmd.setOpCode(opcode);
    cmd.setSyncIndicator(sync_id);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, queue_select);
    cmd.setWord(2U, uop_count);
    cmd.setWord(3U, aux0);
    cmd.setWord(4U, aux1);
    cmd.launchCmd();
}

int
main(void)
{
    launch_cmd(SEU_OP_EXEC, 0U, 3U, 100000U, 0U, 0x50U);
    launch_cmd(SEU_OP_LOAD, 0U, 2048U, 0U, 4U, 0x51U);
    launch_cmd(SEU_OP_STORE, 1U, 2048U, 0U, 4U, 0x52U);
    launch_cmd(SEU_OP_LOAD, 1U, 1024U, 8U, 4U, 0x53U);
    launch_cmd(SEU_OP_EXEC, 0U, 1U, 150000U, 0U, 0x54U);
    launch_cmd(SEU_OP_STORE, 0U, 3072U, 16U, 4U, 0x55U);
    launch_cmd(SEU_OP_LOAD, 0U, 4096U, 32U, 4U, 0x56U);
    launch_cmd(SEU_OP_EXEC, 0U, 2U, 50000U, 0U, 0x57U);
    launch_cmd(SEU_OP_STORE, 1U, 4096U, 16U, 4U, 0x58U);
    launch_cmd(SEU_OP_EXEC, 0U, 4U, 25000U, 0U, 0x59U);

    npu_launch_sync_wait(0U, 0x50U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x51U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x52U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x53U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x54U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x55U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x56U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x57U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x58U, 0U, 0U, 0U);
    npu_launch_sync_wait(0U, 0x59U, 0U, 0U, 0U);
    return 0;
}
