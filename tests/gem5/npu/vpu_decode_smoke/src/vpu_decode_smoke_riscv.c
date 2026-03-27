/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include "vpu.hh"

enum DecodeSmokeLayout
{
    VPU_DEVICE_ID = 0U,
    VPU_SLOT_STRIDE_BYTES = 0x40U,
    LEGACY_SYNC = 0x31U,
    VADD_SYNC = 0x32U,
    SRC0_PORT = 1U,
    SRC1_PORT = 2U,
    DST_PORT = 3U,
    ELEM_COUNT = 4U,
};

static volatile uint32_t *
slot_word_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        0x60000000UL + ((uint64_t)port_id * VPU_SLOT_STRIDE_BYTES));
}

static uint32_t
legacy_mix(uint32_t current, uint32_t signature, uint32_t iteration,
           uint32_t port_id)
{
    return current + signature + ((iteration + 1U) * 0x10U) + (port_id + 1U);
}

static void
clear_slot(uint32_t port_id)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U; idx < VPU_SLOT_STRIDE_BYTES / sizeof(uint32_t);
         ++idx) {
        base[idx] = 0U;
    }
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
wait_slot_word(uint32_t port_id, uint32_t expected, uint64_t timeout)
{
    volatile uint32_t *slot = slot_word_ptr(port_id);

    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        if (*slot == expected) {
            return 0;
        }
    }

    return -1;
}

static int
wait_vector_match(uint32_t port_id, const uint32_t *expected,
                  uint32_t count, uint64_t timeout)
{
    uint32_t actual[ELEM_COUNT];

    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            actual[idx] = slot_word_ptr(port_id)[idx];
            if (actual[idx] != expected[idx]) {
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
    const uint32_t legacy_seed = 0x01020304U;
    const uint32_t src0[ELEM_COUNT] = {1U, 3U, 5U, 7U};
    const uint32_t src1[ELEM_COUNT] = {2U, 4U, 6U, 8U};
    uint32_t expected_vadd[ELEM_COUNT];
    uint32_t actual_vadd[ELEM_COUNT];

    clear_slot(0U);
    clear_slot(SRC0_PORT);
    clear_slot(SRC1_PORT);
    clear_slot(DST_PORT);

    *slot_word_ptr(0U) = legacy_seed;
    vpu_cmd_launch_legacy_exec(VPU_DEVICE_ID, LEGACY_SYNC, 0x1U, 0x1U, 1U);

    if (wait_slot_word(0U, legacy_mix(legacy_seed, legacy_seed, 0U, 0U),
                       60000000ULL) != 0) {
        printf("VPU_DECODE_SMOKE_FAIL legacy actual=%#x\n", *slot_word_ptr(0U));
        return 1;
    }

    store_u32_vector(SRC0_PORT, src0, ELEM_COUNT);
    store_u32_vector(SRC1_PORT, src1, ELEM_COUNT);

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected_vadd[idx] = src0[idx] + src1[idx];
    }

    vpu_cmd_launch_binary(VPU_DEVICE_ID, VPU_OP_VADD, VADD_SYNC, 0x6U, 0x8U,
                          1U, ELEM_COUNT, sizeof(uint32_t),
                          sizeof(uint32_t), VPU_DATA_I32);

    if (wait_vector_match(DST_PORT, expected_vadd, ELEM_COUNT,
                          60000000ULL) != 0) {
        load_u32_vector(DST_PORT, actual_vadd, ELEM_COUNT);
        printf("VPU_DECODE_SMOKE_FAIL vector");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%u act[%u]=%u", idx, expected_vadd[idx], idx,
                   actual_vadd[idx]);
        }
        printf("\n");
        return 1;
    }

    printf("VPU_DECODE_SMOKE_PASS\n");
    return 0;
}
