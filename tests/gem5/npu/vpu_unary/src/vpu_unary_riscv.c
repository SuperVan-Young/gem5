/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "golden/vpu_unary.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum UnaryLayout
{
    VPU_DEVICE_ID = 0U,
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

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src_bits[idx], &src[idx], sizeof(src_bits[idx]));
    }
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);

    /*
     * The VPU scale command repeats the same operation count times, but each
     * repetition reads the original source vector again. The expected result
     * therefore only applies the scalar once.
     */
    npu_golden_vpu_unary_scale_i32(src, expected_i32, ELEM_COUNT, 1U, scalar);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&expected[idx], &expected_i32[idx], sizeof(expected[idx]));
    }

    vpu_cmd_launch_scale(VPU_DEVICE_ID, INT_SCALE_SYNC, 0x1U, 0x1U,
                         repetition, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_I32, (uint32_t)scalar);

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(SRC_PORT),
                                  expected, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(SRC_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL int_scale\n");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf("  exp[%u]=%d act[%u]=%d\n", idx, expected_i32[idx], idx,
                   (int32_t)actual[idx]);
        }
        return -1;
    }

    return 0;
}

static int
run_fp_scale_case(uint32_t repetition, float scalar)
{
    const uint32_t src[ELEM_COUNT] = {
        npu_float_to_bits(1.5f),
        npu_float_to_bits(-2.0f),
        npu_float_to_bits(4.0f),
        npu_float_to_bits(8.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];
    const uint32_t scalar_bits = npu_float_to_bits(scalar);

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_unary_scale_f32(src, expected, ELEM_COUNT, 1U,
                                   scalar_bits);

    vpu_cmd_launch_scale(VPU_DEVICE_ID, FP_SCALE_SYNC, 0x1U, 0x1U,
                         repetition, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32, scalar_bits);

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(SRC_PORT),
                                  expected, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(SRC_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL fp_scale\n");
        npu_expect_u32_vector("VPU_UNARY_FP_SCALE", expected, actual,
                              ELEM_COUNT);
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

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src[idx], &src_i32[idx], sizeof(src[idx]));
    }
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_unary_i2f(src_i32, expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VCVT_I2F, I2F_SYNC, 0x1U, 0x2U,
                         1U, ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
                         VPU_DATA_F32);

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(DST_PORT),
                                  expected, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL i2f\n");
        npu_expect_u32_vector("VPU_UNARY_I2F", expected, actual, ELEM_COUNT);
        return -1;
    }

    return 0;
}

static int
run_f2i_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        npu_float_to_bits(1.75f),
        npu_float_to_bits(-2.5f),
        npu_float_to_bits(3.0f),
        npu_float_to_bits(-0.25f),
    };
    int32_t expected_i32[ELEM_COUNT];
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_unary_f2i(src, expected_i32, ELEM_COUNT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&expected[idx], &expected_i32[idx], sizeof(expected[idx]));
    }

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VCVT_F2I, F2I_SYNC, 0x1U, 0x2U,
                         1U, ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
                         VPU_DATA_I32);

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(DST_PORT),
                                  expected, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL f2i\n");
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf("  exp[%u]=%d act[%u]=%d\n", idx, expected_i32[idx], idx,
                   (int32_t)actual[idx]);
        }
        return -1;
    }

    return 0;
}

static int
run_sqrt_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        npu_float_to_bits(1.0f),
        npu_float_to_bits(4.0f),
        npu_float_to_bits(9.0f),
        npu_float_to_bits(16.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_unary_sqrt(src, expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VSQRT, SQRT_SYNC, 0x1U, 0x2U,
                         1U, ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
                         VPU_DATA_F32);

    if (npu_wait_float_vector_close(NULL, npu_spm_slot_word_ptr_default(DST_PORT),
                                    expected, ELEM_COUNT, 60000000ULL, 0.02f,
                                    0.02f) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL sqrt\n");
        npu_expect_float_vector_close("VPU_UNARY_SQRT", expected, actual,
                                      ELEM_COUNT, 0.02f, 0.02f);
        return -1;
    }

    return 0;
}

static int
run_exp_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        npu_float_to_bits(-1.0f),
        npu_float_to_bits(-0.5f),
        npu_float_to_bits(0.0f),
        npu_float_to_bits(1.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_unary_exp(src, expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VEXP, EXP_SYNC, 0x1U, 0x2U,
                         1U, ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
                         VPU_DATA_F32);

    if (npu_wait_float_vector_close(NULL, npu_spm_slot_word_ptr_default(DST_PORT),
                                    expected, ELEM_COUNT, 60000000ULL, 0.03f,
                                    0.03f) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_FAIL exp\n");
        npu_expect_float_vector_close("VPU_UNARY_EXP", expected, actual,
                                      ELEM_COUNT, 0.03f, 0.03f);
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
