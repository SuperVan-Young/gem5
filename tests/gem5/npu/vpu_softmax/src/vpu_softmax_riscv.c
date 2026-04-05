/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vpu.hh"

enum SoftmaxLayout
{
    VPU_DEVICE_ID = 0U,
    VPU_SLOT_STRIDE_BYTES = 0x40U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SOFTMAX_SYNC0 = 0x81U,
    SOFTMAX_SYNC1 = 0x82U,
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
float_close(float expected, float actual, float abs_tol, float rel_tol)
{
    const float diff = fabsf(expected - actual);
    const float limit = abs_tol + (rel_tol * fabsf(expected));
    return diff <= limit;
}

static int
wait_float_vector_close(uint32_t port_id, const uint32_t *expected,
                        uint32_t count, uint64_t timeout, float abs_tol,
                        float rel_tol)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            const float expected_value = bits_to_float(expected[idx]);
            const float actual_value = bits_to_float(slot_word_ptr(port_id)[idx]);
            if (!float_close(expected_value, actual_value, abs_tol, rel_tol)) {
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
accumulate_expected_stats(struct VpuStatsExpectation *stats)
{
    stats->completed_cmds += 1U;
    stats->prologues += 1U;
    stats->executes += 1U;
    stats->epilogues += 1U;
    stats->read_resps += 1U;
    stats->write_resps += 1U;
    stats->iterations += 1U;
}

static void
compute_softmax_expected(const uint32_t *src, uint32_t *expected)
{
    float max_value = -INFINITY;
    float sum = 0.0f;
    float exp_values[ELEM_COUNT];

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        max_value = fmaxf(max_value, bits_to_float(src[idx]));
    }

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        exp_values[idx] = expf(bits_to_float(src[idx]) - max_value);
        sum += exp_values[idx];
    }

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected[idx] = float_to_bits(exp_values[idx] / sum);
    }
}

static int
run_softmax_case(const uint32_t *src, uint32_t sync_indicator,
                 const char *tag)
{
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    compute_softmax_expected(src, expected);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VSOFTMAX, sync_indicator,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (wait_float_vector_close(DST_PORT, expected, ELEM_COUNT, 80000000ULL,
                                0.03f, 0.03f) != 0) {
        load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_SOFTMAX_FAIL %s", tag);
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
        0x3f800000U, 0x40000000U, 0x40400000U, 0x40800000U,
    };
    const uint32_t src1[ELEM_COUNT] = {
        0x40000000U, 0x40000000U, 0x40000000U, 0x40000000U,
    };
    struct VpuStatsExpectation stats = {0};

    if (run_softmax_case(src0, SOFTMAX_SYNC0, "ramp") != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats);

    if (run_softmax_case(src1, SOFTMAX_SYNC1, "flat") != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats);

    printf("VPU_SOFTMAX_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_SOFTMAX_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_SOFTMAX_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_SOFTMAX_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_SOFTMAX_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_SOFTMAX_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_SOFTMAX_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_SOFTMAX_PASS\n");
    return 0;
}
