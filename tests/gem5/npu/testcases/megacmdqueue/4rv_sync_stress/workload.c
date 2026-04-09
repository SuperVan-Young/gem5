#include <stdint.h>
#include <stdlib.h>

#include "cmd/common.hh"
#include "npu_sync.hh"

enum VpuOpcode
{
    VPU_OP_EXEC = 0x0U,
};

#define DEVICE_ID 0x0U
#define SEU_CMD_READ_MASK 0x00000000U
#define SEU_CMD_WRITE_MASK 0x00000000U
#define SEU_CMD_REPETITION 0x00000001U
#define SEU_CMD_RESERVED 0x00000000U

static void
launch_vpu_exec_at(uint64_t port_base, uint32_t sync_indicator)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(DEVICE_ID);
    cmd.setOpCode(VPU_OP_EXEC);
    cmd.setSyncIndicator(sync_indicator);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, SEU_CMD_READ_MASK);
    cmd.setWord(2U, SEU_CMD_WRITE_MASK);
    cmd.setWord(3U, SEU_CMD_REPETITION);
    cmd.setWord(4U, SEU_CMD_RESERVED);
    cmd.launchCmdAt(port_base);
}

int
main(int argc, char **argv)
{
    uint32_t cpu_id = 0;
    uint32_t rounds = 64;

    if (argc > 1) {
        cpu_id = (uint32_t)strtoul(argv[1], NULL, 0);
    }
    if (argc > 2) {
        uint32_t parsed = (uint32_t)strtoul(argv[2], NULL, 0);
        if (parsed > 0) {
            rounds = parsed;
        }
    }

    const uint64_t port_base = NPU_CMD_PORT_BASE_FOR(cpu_id);
    const uint32_t sync_idx = cpu_id & 0xFFU;

    for (uint32_t round = 0; round < rounds; ++round) {
        npu_launch_sync_wait_at(
            DEVICE_ID,
            sync_idx,
            0x10000000U | (cpu_id << 8) | (round & 0xFFU),
            0x20000000U | (cpu_id << 8) | (round & 0xFFU),
            0x30000000U | (cpu_id << 8) | (round & 0xFFU),
            port_base
        );

        launch_vpu_exec_at(
            port_base,
            ((cpu_id & 0xFFU) << 8) | (round & 0xFFU)
        );

        npu_sync_signal_set(DEVICE_ID, sync_idx);

        volatile uint32_t jitter = 0;
        for (uint32_t i = 0; i < ((cpu_id + 1U) * 16U); ++i) {
            jitter += (i ^ round);
        }
        (void)jitter;
    }

    npu_cmd_sync_done_at(port_base);
    return 0;
}
