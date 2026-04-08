#include <stdint.h>

#include "cmd/common.hh"
#include "npu_mmio.hh"

enum MegaOpcode
{
    MEGA_OP_ENQUEUE = 0x0U,
};

static void
launch_mega_cmd(uint32_t tag)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_MEGA_CMD_QUEUE);
    cmd.setDeviceId(0U);
    cmd.setOpCode(MEGA_OP_ENQUEUE);
    cmd.setSyncIndicator(0U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, tag | 0x1U);
    cmd.setWord(2U, tag | 0x2U);
    cmd.setWord(3U, tag | 0x3U);
    cmd.launchCmdViaMmio();
}

int
main(void)
{
    launch_mega_cmd(0x1000U);
    launch_mega_cmd(0x2000U);

    npu_mmio_write32_one(NPU_CMD_CTRL_ADDR(NPU_CMD_PORT_BASE), 1U);

    launch_mega_cmd(0x3000U);
    return 0;
}
