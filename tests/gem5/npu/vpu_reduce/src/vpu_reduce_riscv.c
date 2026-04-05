/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vpu.hh"

enum ReduceLayout
{
    VPU_DEVICE_ID = 0U,
    VPU_SLOT_STRIDE_BYTES = 0x40U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    INT_SUM_SYNC = 0x71U,
    FP_SUM_SYNC = 0x72U,
    FP_MAX_SYNC = 0x73U,
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

static int
wait_scalar_match(uint32_t port_id, uint32_t expected, uint64_t timeout)
{
    volatile uint32_t *slot = slot_word_ptr(port_id);

    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        if (*slot == expected) {
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
    stats->read_resps += repetition;
    stats->write_resps += repetition;
    stats->iterations += repetition;
}

static int
run_int_reduce_sum_case(void)
{
    const int32_t src_i32[ELEM_COUNT] = {2, -3, 4, 7};
    uint32_t src[ELEM_COUNT];
    int32_t sum = 0;
    uint32_t expected = 0;

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src[idx], &src_i32[idx], sizeof(uint32_t));
        sum += src_i32[idx];
    }
    memcpy(&expected, &sum, sizeof(uint32_t));
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VREDUCE_SUM, INT_SUM_SYNC,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_I32);

    if (wait_scalar_match(DST_PORT, expected, 60000000ULL) != 0) {
        printf("VPU_REDUCE_FAIL int_sum exp=%d act=%d\n", sum,
               (int32_t)*slot_word_ptr(DST_PORT));
        return -1;
    }

    return 0;
}

static int
run_fp_reduce_sum_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        float_to_bits(1.5f),
        float_to_bits(-2.0f),
        float_to_bits(4.0f),
        float_to_bits(0.25f),
    };
    float sum = 0.0f;
    uint32_t expected = 0;

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        sum += bits_to_float(src[idx]);
    }
    expected = float_to_bits(sum);
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VREDUCE_SUM, FP_SUM_SYNC,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (wait_scalar_match(DST_PORT, expected, 60000000ULL) != 0) {
        printf("VPU_REDUCE_FAIL fp_sum exp=%#x act=%#x\n", expected,
               *slot_word_ptr(DST_PORT));
        return -1;
    }

    return 0;
}

static int
run_fp_reduce_max_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        float_to_bits(-3.0f),
        float_to_bits(5.5f),
        float_to_bits(2.25f),
        float_to_bits(4.0f),
    };
    float current_max = -INFINITY;
    uint32_t expected = 0;

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        current_max = fmaxf(current_max, bits_to_float(src[idx]));
    }
    expected = float_to_bits(current_max);
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VREDUCE_MAX, FP_MAX_SYNC,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (wait_scalar_match(DST_PORT, expected, 60000000ULL) != 0) {
        printf("VPU_REDUCE_FAIL fp_max exp=%#x act=%#x\n", expected,
               *slot_word_ptr(DST_PORT));
        return -1;
    }

    return 0;
}

int
main(void)
{
    struct VpuStatsExpectation stats = {0};

    if (run_int_reduce_sum_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_fp_reduce_sum_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_fp_reduce_max_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    printf("VPU_REDUCE_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_REDUCE_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_REDUCE_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_REDUCE_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_REDUCE_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_REDUCE_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_REDUCE_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_REDUCE_PASS\n");
    return 0;
}
