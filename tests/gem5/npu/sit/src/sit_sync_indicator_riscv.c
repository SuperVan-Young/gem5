#include <stdint.h>

#include "cmd/common.hh"
#include "npu_sync.hh"

enum VpuOpcode {
    VPU_OP_EXEC = 0x0U,
};

#define DEVICE_ID 0x0U
#define SYNC_INDEX 7U

int
main(void)
{
    NpuCmd cmd;

    npu_launch_sync_wait(DEVICE_ID, SYNC_INDEX, 0x11111111U, 0x22222222U,
                         0x33333333U);

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(DEVICE_ID);
    cmd.setOpCode(VPU_OP_EXEC);
    cmd.setSyncIndicator(0x55AAU);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, 0xAAA00001U);
    cmd.setWord(2U, 0xBBB00002U);
    cmd.setWord(3U, 0xCCC00003U);
    cmd.launchCmd();

    npu_sync_signal_set(DEVICE_ID, SYNC_INDEX);

    for (;;) {
        asm volatile("" ::: "memory");
    }
}
