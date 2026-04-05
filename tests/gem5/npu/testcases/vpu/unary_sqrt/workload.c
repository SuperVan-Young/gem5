/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "golden/vpu_unary.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum UnarySqrtLayout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x55U,
};

int
main(void)
{
    const uint32_t src_bits[ELEM_COUNT] = {
        npu_float_to_bits(1.0f),
        npu_float_to_bits(4.0f),
        npu_float_to_bits(9.0f),
        npu_float_to_bits(16.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_unary_sqrt(src_bits, expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VSQRT, SYNC_INDICATOR, 0x1U,
                         0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_float_vector_close(NULL,
                                    npu_spm_slot_word_ptr_default(DST_PORT),
                                    expected, ELEM_COUNT, 60000000ULL, 0.02f,
                                    0.02f) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_SQRT_FAIL\n");
        npu_expect_float_vector_close("VPU_UNARY_SQRT", expected, actual,
                                      ELEM_COUNT, 0.02f, 0.02f);
        return 1;
    }

    printf("VPU_UNARY_SQRT_PASS\n");
    return 0;
}

