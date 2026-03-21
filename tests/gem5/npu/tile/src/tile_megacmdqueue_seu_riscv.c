#include <stdint.h>

#include "cmd/common.hh"

enum VpuOpcode {
    VPU_OP_EXEC = 0x0U,
};

#define DEVICE_ID 0x0U
#define NUM_CMDS 6

static inline void
emit_command(uint32_t seq)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(DEVICE_ID);
    cmd.setOpCode(VPU_OP_EXEC);
    cmd.setSyncIndicator(seq);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, 0xA0000000U | seq);
    cmd.setWord(2U, 0xB0000000U | seq);
    cmd.setWord(3U, 0xC0000000U | seq);
    cmd.launchCmd();
}

int
main(void)
{
    for (uint32_t i = 0; i < NUM_CMDS; ++i) {
        emit_command(i);
    }

    volatile uint64_t spin = 0;
    for (uint64_t i = 0; i < 2000000ULL; ++i) {
        spin += i;
    }

    return (int)(spin & 0);
}
