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

enum UnaryAbsLayout
{
    VPU_DEVICE_ID = 0U,
    F32_SRC_PORT = 0U,
    F32_DST_PORT = 1U,
    I32_SRC_PORT = 2U,
    I32_DST_PORT = 3U,
    ELEM_COUNT = 4U,
    F32_SYNC_INDICATOR = 0x57U,
    I32_SYNC_INDICATOR = 0x58U,
};

static int
run_f32_case(void)
{
    const uint32_t src_bits[ELEM_COUNT] = {
        npu_float_to_bits(-3.5f),
        npu_float_to_bits(-0.0f),
        npu_float_to_bits(0.0f),
        npu_float_to_bits(2.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(F32_SRC_PORT);
    npu_spm_clear_slot(F32_DST_PORT);
    npu_spm_store_u32_vector(F32_SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_unary_abs_f32(src_bits, expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VABS, F32_SYNC_INDICATOR,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_float_vector_close(NULL,
                                    npu_spm_slot_word_ptr_default(
                                        F32_DST_PORT),
                                    expected, ELEM_COUNT, 60000000ULL, 0.0f,
                                    0.0f) != 0) {
        npu_spm_load_u32_vector(F32_DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_ABS_F32_FAIL\n");
        npu_expect_float_vector_close("VPU_UNARY_ABS_F32", expected, actual,
                                      ELEM_COUNT, 0.0f, 0.0f);
        return 1;
    }

    return 0;
}

static int
run_i32_case(void)
{
    const int32_t src[ELEM_COUNT] = {-7, -1, 0, 9};
    int32_t expected_i32[ELEM_COUNT];
    uint32_t src_bits[ELEM_COUNT];
    uint32_t expected_bits[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src_bits[idx], &src[idx], sizeof(src_bits[idx]));
    }

    npu_spm_clear_slot(I32_SRC_PORT);
    npu_spm_clear_slot(I32_DST_PORT);
    npu_spm_store_u32_vector(I32_SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_unary_abs_i32(src, expected_i32, ELEM_COUNT);

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&expected_bits[idx], &expected_i32[idx],
               sizeof(expected_bits[idx]));
    }

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VABS, I32_SYNC_INDICATOR,
                         0x4U, 0x8U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_I32);

    if (npu_wait_u32_vector_match(NULL,
                                  npu_spm_slot_word_ptr_default(I32_DST_PORT),
                                  expected_bits, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(I32_DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_ABS_I32_FAIL\n");
        npu_expect_u32_vector("VPU_UNARY_ABS_I32", expected_bits, actual,
                              ELEM_COUNT);
        return 1;
    }

    return 0;
}

int
main(void)
{
    if (run_f32_case() != 0) {
        return 1;
    }
    if (run_i32_case() != 0) {
        return 1;
    }

    printf("VPU_UNARY_ABS_PASS\n");
    return 0;
}
