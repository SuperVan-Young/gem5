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

enum UnaryF2ILayout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x54U,
};

int
main(void)
{
    const uint32_t src_bits[ELEM_COUNT] = {
        npu_float_to_bits(1.75f),
        npu_float_to_bits(-2.5f),
        npu_float_to_bits(3.0f),
        npu_float_to_bits(-0.25f),
    };
    int32_t expected_i32[ELEM_COUNT];
    uint32_t expected_bits[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_unary_f2i(src_bits, expected_i32, ELEM_COUNT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&expected_bits[idx], &expected_i32[idx],
               sizeof(expected_bits[idx]));
    }

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VCVT_F2I, SYNC_INDICATOR,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_I32);

    const volatile uint32_t *expected_slot =
        npu_spm_slot_word_ptr_default(DST_PORT);

    if (npu_wait_u32_vector_match(NULL, expected_slot, expected_bits,
                                  ELEM_COUNT, 60000000ULL) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_F2I_FAIL\n");
        npu_expect_u32_vector("VPU_UNARY_F2I", expected_bits, actual,
                              ELEM_COUNT);
        return 1;
    }

    printf("VPU_UNARY_F2I_PASS\n");
    return 0;
}
