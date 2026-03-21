#include <stdint.h>

#include "cmd/common.hh"
#include "npu_sync.hh"

enum VpuOpcode {
    VPU_OP_EXEC = 0x0U,
    VPU_OP_INDICATOR_SET = 0x1U,
};

#define DEVICE_ID 0x0U
#define SYNC_IDX0 7U
#define SYNC_IDX1 9U

static void
launch_indicator_set(uint32_t sync_idx, uint32_t w1, uint32_t w2, uint32_t w3)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(DEVICE_ID);
    cmd.setOpCode(VPU_OP_INDICATOR_SET);
    cmd.setSyncIndicator(sync_idx);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, w1);
    cmd.setWord(2U, w2);
    cmd.setWord(3U, w3);
    cmd.launchCmd();
}

int
main(void)
{
    launch_indicator_set(SYNC_IDX0, 0xA0010001U, 0xA0010002U, 0xA0010003U);
    npu_launch_sync_wait(DEVICE_ID, SYNC_IDX0, 0xB0010001U, 0xB0010002U,
                         0xB0010003U);

    launch_indicator_set(SYNC_IDX1, 0xA0020001U, 0xA0020002U, 0xA0020003U);
    npu_launch_sync_wait(DEVICE_ID, SYNC_IDX1, 0xB0020001U, 0xB0020002U,
                         0xB0020003U);

    for (;;) {
        asm volatile("" ::: "memory");
    }
}
