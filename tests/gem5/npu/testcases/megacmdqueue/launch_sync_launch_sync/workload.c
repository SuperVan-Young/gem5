#include <stdint.h>

#include "cmd/common.hh"
#include "npu_sync.hh"

enum VpuOpcode
{
    VPU_OP_INDICATOR_SET = 0x1U,
};

#define DEVICE_ID 0x0U
#define SYNC_IDX0 7U
#define SYNC_IDX1 9U
#define SEU_CMD_READ_MASK 0x00000000U
#define SEU_CMD_WRITE_MASK 0x00000000U
#define SEU_CMD_REPETITION 0x00000001U
#define SEU_CMD_RESERVED 0x00000000U

static void
launch_indicator_set(uint32_t sync_idx)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(DEVICE_ID);
    cmd.setOpCode(VPU_OP_INDICATOR_SET);
    cmd.setSyncIndicator(sync_idx);
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
    launch_indicator_set(SYNC_IDX0);
    npu_launch_sync_wait(
        DEVICE_ID,
        SYNC_IDX0,
        0xB0010001U,
        0xB0010002U,
        0xB0010003U
    );

    launch_indicator_set(SYNC_IDX1);
    npu_launch_sync_wait(
        DEVICE_ID,
        SYNC_IDX1,
        0xB0020001U,
        0xB0020002U,
        0xB0020003U
    );

    npu_cmd_sync_done();
    return 0;
}
