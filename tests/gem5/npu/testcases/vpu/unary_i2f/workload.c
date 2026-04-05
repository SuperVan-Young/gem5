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

enum UnaryI2FLayout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x53U,
};

int
main(void)
{
    const int32_t src_i32[ELEM_COUNT] = {1, -2, 7, -9};
    uint32_t src_bits[ELEM_COUNT];
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src_bits[idx], &src_i32[idx], sizeof(src_bits[idx]));
    }

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_unary_i2f(src_i32, expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VCVT_I2F, SYNC_INDICATOR,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    const volatile uint32_t *expected_slot =
        npu_spm_slot_word_ptr_default(DST_PORT);

    if (npu_wait_u32_vector_match(NULL, expected_slot, expected, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_I2F_FAIL\n");
        npu_expect_u32_vector("VPU_UNARY_I2F", expected, actual, ELEM_COUNT);
        return 1;
    }

    printf("VPU_UNARY_I2F_PASS\n");
    return 0;
}
