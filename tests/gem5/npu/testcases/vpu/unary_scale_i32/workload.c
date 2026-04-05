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

enum UnaryScaleI32Layout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x51U,
};

int
main(void)
{
    const int32_t src[ELEM_COUNT] = {2, -3, 4, -5};
    int32_t expected_i32[ELEM_COUNT];
    uint32_t expected_bits[ELEM_COUNT];
    uint32_t src_bits[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src_bits[idx], &src[idx], sizeof(src_bits[idx]));
    }

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_unary_scale_i32(src, expected_i32, ELEM_COUNT, 1U, 3);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(
            &expected_bits[idx],
            &expected_i32[idx],
            sizeof(expected_bits[idx]));
    }

    vpu_cmd_launch_scale(
        VPU_DEVICE_ID,
        SYNC_INDICATOR,
        0x1U,
        0x1U,
        1U,
        ELEM_COUNT,
        sizeof(uint32_t),
        sizeof(uint32_t),
        VPU_DATA_I32,
        3U);

    const volatile uint32_t *expected_slot =
        npu_spm_slot_word_ptr_default(SRC_PORT);

    if (npu_wait_u32_vector_match(NULL, expected_slot, expected_bits,
                                  ELEM_COUNT, 60000000ULL) != 0) {
        npu_spm_load_u32_vector(SRC_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_SCALE_I32_FAIL\n");
        npu_expect_u32_vector(
            "VPU_UNARY_SCALE_I32", expected_bits, actual, ELEM_COUNT);
        return 1;
    }

    printf("VPU_UNARY_SCALE_I32_PASS\n");
    return 0;
}
