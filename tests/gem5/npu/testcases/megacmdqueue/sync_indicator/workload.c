#include <stdint.h>

#include "cmd/common.hh"
#include "npu_sync.hh"

enum VpuOpcode
{
    VPU_OP_EXEC = 0x0U,
};

#define DEVICE_ID 0x0U
#define SYNC_INDEX 7U
#define SEU_CMD_READ_MASK 0x00000000U
#define SEU_CMD_WRITE_MASK 0x00000000U
#define SEU_CMD_REPETITION 0x00000001U
#define SEU_CMD_RESERVED 0x00000000U

int
main(void)
{
    NpuCmd cmd;

    npu_launch_sync_wait(
        DEVICE_ID,
        SYNC_INDEX,
        0x11111111U,
        0x22222222U,
        0x33333333U
    );

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(DEVICE_ID);
    cmd.setOpCode(VPU_OP_EXEC);
    cmd.setSyncIndicator(0x55AAU);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, SEU_CMD_READ_MASK);
    cmd.setWord(2U, SEU_CMD_WRITE_MASK);
    cmd.setWord(3U, SEU_CMD_REPETITION);
    cmd.setWord(4U, SEU_CMD_RESERVED);
    cmd.launchCmd();

    npu_sync_signal_set(DEVICE_ID, SYNC_INDEX);
    npu_cmd_sync_done();
    return 0;
}
