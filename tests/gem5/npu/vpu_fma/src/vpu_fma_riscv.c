/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vpu.hh"

enum FmaLayout
{
    VPU_DEVICE_ID = 0U,
    VPU_SLOT_STRIDE_BYTES = 0x40U,
    SRC0_PORT = 0U,
    SRC1_PORT = 1U,
    SRC2_PORT = 2U,
    DST_PORT = 3U,
    ELEM_COUNT = 4U,
    FMA_SYNC = 0x61U,
    FMA_INPLACE_SYNC = 0x62U,
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

static uint32_t
float_to_bits(float value)
{
    uint32_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float
bits_to_float(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
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

static void
accumulate_expected_stats(struct VpuStatsExpectation *stats,
                          uint32_t repetition)
{
    stats->completed_cmds += 1U;
    stats->prologues += repetition;
    stats->executes += repetition;
    stats->epilogues += repetition;
    stats->read_resps += 3U * repetition;
    stats->write_resps += repetition;
    stats->iterations += repetition;
}

static int
run_fma_case(uint32_t sync_indicator, const uint32_t *src0,
             const uint32_t *src1, const uint32_t *src2, uint32_t *expected,
             uint32_t repetition, uint32_t dst_port)
{
    uint32_t actual[ELEM_COUNT];
    const uint32_t read_mask = (1U << SRC0_PORT) | (1U << SRC1_PORT) |
                               (1U << SRC2_PORT);
    const uint32_t write_mask = 1U << dst_port;

    clear_slot(SRC0_PORT);
    clear_slot(SRC1_PORT);
    clear_slot(SRC2_PORT);
    clear_slot(DST_PORT);
    store_u32_vector(SRC0_PORT, src0, ELEM_COUNT);
    store_u32_vector(SRC1_PORT, src1, ELEM_COUNT);
    store_u32_vector(SRC2_PORT, src2, ELEM_COUNT);

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        const float lhs = bits_to_float(src0[idx]);
        const float rhs = bits_to_float(src1[idx]);
        const float acc = bits_to_float(src2[idx]);
        expected[idx] = float_to_bits(fmaf(lhs, rhs, acc));
    }

    vpu_cmd_launch_ternary(VPU_DEVICE_ID, VPU_OP_VFMA, sync_indicator,
                           read_mask, write_mask, repetition, ELEM_COUNT,
                           sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32);

    if (wait_vector_match(dst_port, expected, ELEM_COUNT, 60000000ULL) != 0) {
        load_u32_vector(dst_port, actual, ELEM_COUNT);
        printf("VPU_FMA_FAIL");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%#x act[%u]=%#x", idx, expected[idx], idx,
                   actual[idx]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

int
main(void)
{
    const uint32_t src0[ELEM_COUNT] = {
        float_to_bits(1.0f),
        float_to_bits(-2.0f),
        float_to_bits(0.5f),
        float_to_bits(4.0f),
    };
    const uint32_t src1[ELEM_COUNT] = {
        float_to_bits(3.0f),
        float_to_bits(0.25f),
        float_to_bits(-8.0f),
        float_to_bits(2.0f),
    };
    const uint32_t src2[ELEM_COUNT] = {
        float_to_bits(0.5f),
        float_to_bits(1.0f),
        float_to_bits(2.0f),
        float_to_bits(-1.0f),
    };
    uint32_t expected[ELEM_COUNT];
    struct VpuStatsExpectation stats = {0};

    if (run_fma_case(FMA_SYNC, src0, src1, src2, expected, 1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_fma_case(FMA_INPLACE_SYNC, src0, src1, src2, expected, 2U,
                     SRC0_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 2U);

    printf("VPU_FMA_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_FMA_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_FMA_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_FMA_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_FMA_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_FMA_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_FMA_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_FMA_PASS\n");
    return 0;
}
