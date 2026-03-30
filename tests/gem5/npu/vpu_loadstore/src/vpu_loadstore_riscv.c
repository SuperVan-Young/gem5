/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "npu_sync.hh"
#include "vpu.hh"

enum LoadStoreLayout
{
    VPU_DEVICE_ID = 0U,
    VPU_SLOT_STRIDE_BYTES = 0x40U,
    PORT0 = 0U,
    PORT1 = 1U,
    ELEM_COUNT = 4U,
    VLOAD_SYNC = 0x81U,
    VSTORE_SYNC = 0x82U,
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
slot_word_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        0x60000000UL + ((uint64_t)port_id * VPU_SLOT_STRIDE_BYTES));
}

static void
store_u32_vector(uint32_t port_id, const uint32_t *values, uint32_t count)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U; idx < count; ++idx) {
        base[idx] = values[idx];
    }
}

static void
load_u32_vector(uint32_t port_id, uint32_t *values, uint32_t count)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U; idx < count; ++idx) {
        values[idx] = base[idx];
    }
}

static int
wait_vector_match(uint32_t port_id, const uint32_t *expected,
                  uint32_t count, uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            if (slot_word_ptr(port_id)[idx] != expected[idx]) {
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
    const uint32_t src0[ELEM_COUNT] = {11U, 22U, 33U, 44U};
    const uint32_t src1[ELEM_COUNT] = {101U, 202U, 303U, 404U};
    const uint32_t poison0[ELEM_COUNT] = {0xdead0001U, 0xdead0002U,
                                          0xdead0003U, 0xdead0004U};
    const uint32_t poison1[ELEM_COUNT] = {0xbeef0001U, 0xbeef0002U,
                                          0xbeef0003U, 0xbeef0004U};
    uint32_t actual[ELEM_COUNT];
    struct VpuStatsExpectation stats = {2U, 2U, 2U, 2U, 2U, 2U, 2U};

    store_u32_vector(PORT0, src0, ELEM_COUNT);
    store_u32_vector(PORT1, src1, ELEM_COUNT);

    vpu_cmd_launch_load(VPU_DEVICE_ID, VLOAD_SYNC, 0x3U, ELEM_COUNT,
                        sizeof(uint32_t), VPU_DATA_I32);
    npu_launch_sync_wait(VPU_DEVICE_ID, VLOAD_SYNC, 0U, 0U, 0U);
    npu_cmd_sync_done();

    store_u32_vector(PORT0, poison0, ELEM_COUNT);
    store_u32_vector(PORT1, poison1, ELEM_COUNT);

    vpu_cmd_launch_store(VPU_DEVICE_ID, VSTORE_SYNC, 0x3U, ELEM_COUNT,
                         sizeof(uint32_t), VPU_DATA_I32);

    if (wait_vector_match(PORT0, src0, ELEM_COUNT, 60000000ULL) != 0) {
        load_u32_vector(PORT0, actual, ELEM_COUNT);
        printf("VPU_LOADSTORE_FAIL port0");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%u act[%u]=%u", idx, src0[idx], idx, actual[idx]);
        }
        printf("\n");
        return 1;
    }

    if (wait_vector_match(PORT1, src1, ELEM_COUNT, 60000000ULL) != 0) {
        load_u32_vector(PORT1, actual, ELEM_COUNT);
        printf("VPU_LOADSTORE_FAIL port1");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%u act[%u]=%u", idx, src1[idx], idx, actual[idx]);
        }
        printf("\n");
        return 1;
    }

    printf("VPU_LOADSTORE_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_LOADSTORE_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_LOADSTORE_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_LOADSTORE_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_LOADSTORE_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_LOADSTORE_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_LOADSTORE_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_LOADSTORE_PASS\n");
    return 0;
}
