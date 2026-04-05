/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vpu.hh"

enum UnaryLayout
{
    VPU_DEVICE_ID = 0U,
    VPU_SLOT_STRIDE_BYTES = 0x40U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    INT_SCALE_SYNC = 0x51U,
    FP_SCALE_SYNC = 0x52U,
    I2F_SYNC = 0x53U,
    F2I_SYNC = 0x54U,
    SQRT_SYNC = 0x55U,
    EXP_SYNC = 0x56U,
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
run_int_scale_case(uint32_t repetition, int32_t scalar)
{
    const int32_t src[ELEM_COUNT] = {2, -3, 4, -5};
    uint32_t src_bits[ELEM_COUNT];
    int32_t expected_i32[ELEM_COUNT];
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src_bits[idx], &src[idx], sizeof(uint32_t));
    }
    store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected_i32[idx] = src[idx];
    }
    for (uint32_t iter = 0U; iter < repetition; ++iter) {
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            expected_i32[idx] *= scalar;
        }
    }
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&expected[idx], &expected_i32[idx], sizeof(uint32_t));
    }

    vpu_cmd_launch_scale(VPU_DEVICE_ID, INT_SCALE_SYNC, 0x1U, 0x1U,
                         repetition, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_I32,
                         (uint32_t)scalar);

    if (wait_vector_match(SRC_PORT, expected, ELEM_COUNT, 60000000ULL) != 0) {
        load_u32_vector(SRC_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL int_scale");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%d act[%u]=%d", idx, expected_i32[idx], idx,
                   (int32_t)actual[idx]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

static int
run_fp_scale_case(uint32_t repetition, float scalar)
{
    const uint32_t src[ELEM_COUNT] = {
        float_to_bits(1.5f),
        float_to_bits(-2.0f),
        float_to_bits(4.0f),
        float_to_bits(8.0f),
    };
    float expected_fp[ELEM_COUNT];
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];
    const uint32_t scalar_bits = float_to_bits(scalar);

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected_fp[idx] = bits_to_float(src[idx]);
    }
    for (uint32_t iter = 0U; iter < repetition; ++iter) {
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            expected_fp[idx] *= scalar;
        }
    }
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected[idx] = float_to_bits(expected_fp[idx]);
    }

    vpu_cmd_launch_scale(VPU_DEVICE_ID, FP_SCALE_SYNC, 0x1U, 0x1U,
                         repetition, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32, scalar_bits);

    if (wait_vector_match(SRC_PORT, expected, ELEM_COUNT, 60000000ULL) != 0) {
        load_u32_vector(SRC_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL fp_scale");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%#x act[%u]=%#x", idx, expected[idx], idx,
                   actual[idx]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

static int
run_i2f_case(void)
{
    const int32_t src_i32[ELEM_COUNT] = {1, -2, 7, -9};
    uint32_t src[ELEM_COUNT];
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src[idx], &src_i32[idx], sizeof(uint32_t));
        expected[idx] = float_to_bits((float)src_i32[idx]);
    }
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VCVT_I2F, I2F_SYNC, 0x1U, 0x2U,
                         1U, ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
                         VPU_DATA_F32);

    if (wait_vector_match(DST_PORT, expected, ELEM_COUNT, 60000000ULL) != 0) {
        load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL i2f");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%#x act[%u]=%#x", idx, expected[idx], idx,
                   actual[idx]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

static int
run_f2i_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        float_to_bits(1.75f),
        float_to_bits(-2.5f),
        float_to_bits(3.0f),
        float_to_bits(-0.25f),
    };
    const int32_t expected_i32[ELEM_COUNT] = {1, -2, 3, 0};
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&expected[idx], &expected_i32[idx], sizeof(uint32_t));
    }
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VCVT_F2I, F2I_SYNC, 0x1U, 0x2U,
                         1U, ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
                         VPU_DATA_I32);

    if (wait_vector_match(DST_PORT, expected, ELEM_COUNT, 60000000ULL) != 0) {
        load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL f2i");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%d act[%u]=%d", idx, expected_i32[idx], idx,
                   (int32_t)actual[idx]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

static int
run_sqrt_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        float_to_bits(1.0f),
        float_to_bits(4.0f),
        float_to_bits(9.0f),
        float_to_bits(16.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected[idx] = float_to_bits(sqrtf(bits_to_float(src[idx])));
    }
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VSQRT, SQRT_SYNC, 0x1U, 0x2U,
                         1U, ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
                         VPU_DATA_F32);

    if (wait_float_vector_close(DST_PORT, expected, ELEM_COUNT, 60000000ULL,
                                0.02f, 0.02f) != 0) {
        load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL sqrt");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%#x act[%u]=%#x", idx, expected[idx], idx,
                   actual[idx]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

static int
run_exp_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        float_to_bits(-1.0f),
        float_to_bits(-0.5f),
        float_to_bits(0.0f),
        float_to_bits(1.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    clear_slot(SRC_PORT);
    clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected[idx] = float_to_bits(expf(bits_to_float(src[idx])));
    }
    store_u32_vector(SRC_PORT, src, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VEXP, EXP_SYNC, 0x1U, 0x2U,
                         1U, ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
                         VPU_DATA_F32);

    if (wait_float_vector_close(DST_PORT, expected, ELEM_COUNT, 60000000ULL,
                                0.03f, 0.03f) != 0) {
        load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL exp");
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
    struct VpuStatsExpectation stats = {0};

    if (run_int_scale_case(2U, 3) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 2U);

    if (run_fp_scale_case(2U, 0.5f) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 2U);

    if (run_i2f_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_f2i_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_sqrt_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_exp_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    printf("VPU_UNARY_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_UNARY_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_UNARY_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_UNARY_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_UNARY_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_UNARY_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_UNARY_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_UNARY_PASS\n");
    return 0;
}
