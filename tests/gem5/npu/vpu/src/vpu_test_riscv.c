/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "cmd/common.hh"

enum VpuOpcode
{
    VPU_OP_EXEC = 0x0U,
};

enum VpuLayout
{
    VPU_NUM_PORTS = 4U,
    VPU_SLOT_STRIDE_BYTES = 0x40U,
    VPU0_DEVICE_ID = 0x0U,
    VPU1_DEVICE_ID = 0x1U,
    VPU0_CMD0_SYNC = 0x10U,
    VPU0_CMD1_SYNC = 0x11U,
    VPU1_CMD0_SYNC = 0x20U,
    VPU1_CMD1_SYNC = 0x21U,
    VPU0_CMD0_READ_MASK = 0x1U,
    VPU0_CMD0_WRITE_MASK = 0x1U,
    VPU0_CMD0_REPETITION = 1U,
    VPU0_CMD1_READ_MASK = 0x3U,
    VPU0_CMD1_WRITE_MASK = 0x2U,
    VPU0_CMD1_REPETITION = 2U,
    VPU1_CMD0_READ_MASK = 0x4U,
    VPU1_CMD0_WRITE_MASK = 0x4U,
    VPU1_CMD0_REPETITION = 1U,
    VPU1_CMD1_READ_MASK = 0xCU,
    VPU1_CMD1_WRITE_MASK = 0x8U,
    VPU1_CMD1_REPETITION = 2U,
};

struct VpuStatsExpectation
{
    uint32_t completed_cmds;
    uint32_t prologues;
    uint32_t executes;
    uint32_t epilogues;
    uint32_t read_resps;
    uint32_t write_resps;
    uint32_t iterations;
};

static volatile uint32_t *
spm_slot_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        0x60000000UL + ((uint64_t)port_id * VPU_SLOT_STRIDE_BYTES));
}

static uint32_t
popcount32(uint32_t value)
{
    uint32_t count = 0U;

    while (value != 0U) {
        count += value & 1U;
        value >>= 1U;
    }

    return count;
}

static uint32_t
initial_slot_value(uint32_t port_id)
{
    return 0x00010011U + (port_id * 0x00011111U);
}

static uint32_t
mix_slot_value(uint32_t current, uint32_t signature, uint32_t iteration,
               uint32_t port_id)
{
    return current + signature + ((iteration + 1U) * 0x10U) + (port_id + 1U);
}

static void
seed_spm_slots(uint32_t *slots)
{
    for (uint32_t port = 0U; port < VPU_NUM_PORTS; ++port) {
        const uint32_t value = initial_slot_value(port);
        slots[port] = value;
        *spm_slot_ptr(port) = value;
    }
}

static uint32_t
build_signature(const uint32_t *slots, uint32_t read_mask)
{
    uint32_t signature = 0U;

    for (uint32_t port = 0U; port < VPU_NUM_PORTS; ++port) {
        if ((read_mask & (1U << port)) != 0U) {
            signature += slots[port];
        }
    }

    return signature;
}

static void
advance_expected_slots(uint32_t *slots, uint32_t read_mask,
                       uint32_t write_mask, uint32_t repetition)
{
    for (uint32_t iteration = 0U; iteration < repetition; ++iteration) {
        const uint32_t signature = build_signature(slots, read_mask);

        for (uint32_t port = 0U; port < VPU_NUM_PORTS; ++port) {
            if ((write_mask & (1U << port)) != 0U) {
                slots[port] = mix_slot_value(slots[port], signature, iteration,
                                             port);
            }
        }
    }
}

static void
accumulate_expected_stats(struct VpuStatsExpectation *stats,
                          uint32_t read_mask, uint32_t write_mask,
                          uint32_t repetition)
{
    stats->completed_cmds += 1U;
    stats->prologues += repetition;
    stats->executes += repetition;
    stats->epilogues += repetition;
    stats->read_resps += popcount32(read_mask) * repetition;
    stats->write_resps += popcount32(write_mask) * repetition;
    stats->iterations += repetition;
}

static void
launch_vpu_cmd(uint32_t device_id, uint32_t sync_indicator,
               uint32_t read_mask, uint32_t write_mask,
               uint32_t repetition)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(device_id);
    cmd.setOpCode(VPU_OP_EXEC);
    cmd.setSyncIndicator(sync_indicator);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, read_mask);
    cmd.setWord(2U, write_mask);
    cmd.setWord(3U, repetition);
    cmd.setWord(4U, 0U);
    cmd.launchCmd();
}

static void
print_slot_snapshot(const char *prefix, const uint32_t *slots)
{
    printf("%s", prefix);
    for (uint32_t port = 0U; port < VPU_NUM_PORTS; ++port) {
        printf(" P%u=%#x", port, slots[port]);
    }
    printf("\n");
}

static int
wait_for_expected_slots(const uint32_t *expected_slots, uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t port = 0U; port < VPU_NUM_PORTS; ++port) {
            if (*spm_slot_ptr(port) != expected_slots[port]) {
                matched = 0;
                break;
            }
        }

        if (matched) {
            return 0;
        }
    }

    return -1;
}

int
main(void)
{
    uint32_t expected_slots[VPU_NUM_PORTS];
    uint32_t actual_slots[VPU_NUM_PORTS];
    struct VpuStatsExpectation vpu0_stats = {0};
    struct VpuStatsExpectation vpu1_stats = {0};

    seed_spm_slots(expected_slots);
    print_slot_snapshot("VPU_INITIAL", expected_slots);

    launch_vpu_cmd(VPU1_DEVICE_ID, VPU1_CMD0_SYNC, VPU1_CMD0_READ_MASK,
                   VPU1_CMD0_WRITE_MASK, VPU1_CMD0_REPETITION);
    advance_expected_slots(expected_slots, VPU1_CMD0_READ_MASK,
                           VPU1_CMD0_WRITE_MASK, VPU1_CMD0_REPETITION);
    accumulate_expected_stats(&vpu1_stats, VPU1_CMD0_READ_MASK,
                              VPU1_CMD0_WRITE_MASK, VPU1_CMD0_REPETITION);

    launch_vpu_cmd(VPU0_DEVICE_ID, VPU0_CMD0_SYNC, VPU0_CMD0_READ_MASK,
                   VPU0_CMD0_WRITE_MASK, VPU0_CMD0_REPETITION);
    advance_expected_slots(expected_slots, VPU0_CMD0_READ_MASK,
                           VPU0_CMD0_WRITE_MASK, VPU0_CMD0_REPETITION);
    accumulate_expected_stats(&vpu0_stats, VPU0_CMD0_READ_MASK,
                              VPU0_CMD0_WRITE_MASK, VPU0_CMD0_REPETITION);

    launch_vpu_cmd(VPU1_DEVICE_ID, VPU1_CMD1_SYNC, VPU1_CMD1_READ_MASK,
                   VPU1_CMD1_WRITE_MASK, VPU1_CMD1_REPETITION);
    advance_expected_slots(expected_slots, VPU1_CMD1_READ_MASK,
                           VPU1_CMD1_WRITE_MASK, VPU1_CMD1_REPETITION);
    accumulate_expected_stats(&vpu1_stats, VPU1_CMD1_READ_MASK,
                              VPU1_CMD1_WRITE_MASK, VPU1_CMD1_REPETITION);

    launch_vpu_cmd(VPU0_DEVICE_ID, VPU0_CMD1_SYNC, VPU0_CMD1_READ_MASK,
                   VPU0_CMD1_WRITE_MASK, VPU0_CMD1_REPETITION);
    advance_expected_slots(expected_slots, VPU0_CMD1_READ_MASK,
                           VPU0_CMD1_WRITE_MASK, VPU0_CMD1_REPETITION);
    accumulate_expected_stats(&vpu0_stats, VPU0_CMD1_READ_MASK,
                              VPU0_CMD1_WRITE_MASK, VPU0_CMD1_REPETITION);

    if (wait_for_expected_slots(expected_slots, 60000000ULL) != 0) {
        for (uint32_t port = 0U; port < VPU_NUM_PORTS; ++port) {
            actual_slots[port] = *spm_slot_ptr(port);
        }

        print_slot_snapshot("VPU_EXPECTED", expected_slots);
        print_slot_snapshot("VPU_ACTUAL", actual_slots);
        printf("VPU_TEST_FAIL\n");
        return 1;
    }

    print_slot_snapshot("VPU_FINAL", expected_slots);
    printf("VPU0_EXPECTED_COMPLETED_CMDS=%u\n", vpu0_stats.completed_cmds);
    printf("VPU0_EXPECTED_PROLOGUES=%u\n", vpu0_stats.prologues);
    printf("VPU0_EXPECTED_EXECUTES=%u\n", vpu0_stats.executes);
    printf("VPU0_EXPECTED_EPILOGUES=%u\n", vpu0_stats.epilogues);
    printf("VPU0_EXPECTED_READ_RESPS=%u\n", vpu0_stats.read_resps);
    printf("VPU0_EXPECTED_WRITE_RESPS=%u\n", vpu0_stats.write_resps);
    printf("VPU0_EXPECTED_ITERATIONS=%u\n", vpu0_stats.iterations);
    printf("VPU1_EXPECTED_COMPLETED_CMDS=%u\n", vpu1_stats.completed_cmds);
    printf("VPU1_EXPECTED_PROLOGUES=%u\n", vpu1_stats.prologues);
    printf("VPU1_EXPECTED_EXECUTES=%u\n", vpu1_stats.executes);
    printf("VPU1_EXPECTED_EPILOGUES=%u\n", vpu1_stats.epilogues);
    printf("VPU1_EXPECTED_READ_RESPS=%u\n", vpu1_stats.read_resps);
    printf("VPU1_EXPECTED_WRITE_RESPS=%u\n", vpu1_stats.write_resps);
    printf("VPU1_EXPECTED_ITERATIONS=%u\n", vpu1_stats.iterations);
    printf("VPU_TEST_PASS\n");
    return 0;
}
