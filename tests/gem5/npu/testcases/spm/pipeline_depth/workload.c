/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>

#include "cmd/common.hh"
#include "npu_sync.hh"

enum SeuOpcode
{
    SEU_OP_LOAD = 0x1U,
};

static void
launch_load(uint32_t queue_select, uint32_t uop_count, uint32_t sync_id)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(0U);
    cmd.setOpCode(SEU_OP_LOAD);
    cmd.setSyncIndicator(sync_id);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, queue_select);
    cmd.setWord(2U, uop_count);
    cmd.setWord(3U, 0U);
    cmd.setWord(4U, 4U);
    cmd.launchCmd();
}

int
main(void)
{
    launch_load(0U, 256U, 0x70U);
    npu_launch_sync_wait(0U, 0x70U, 0U, 0U, 0U);
    return 0;
}
