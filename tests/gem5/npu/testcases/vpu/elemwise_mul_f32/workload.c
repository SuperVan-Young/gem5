/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "golden/vpu_elemwise.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum ElemwiseMulF32Layout
{
    VPU_DEVICE_ID = 0U,
    SRC0_PORT = 0U,
    SRC1_PORT = 1U,
    DST_PORT = 2U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x47U,
};

int
main(void)
{
    const uint32_t lhs[ELEM_COUNT] = {
        npu_float_to_bits(1.25f),
        npu_float_to_bits(2.5f),
        npu_float_to_bits(3.75f),
        npu_float_to_bits(5.0f),
    };
    const uint32_t rhs[ELEM_COUNT] = {
        npu_float_to_bits(2.0f),
        npu_float_to_bits(3.0f),
        npu_float_to_bits(4.0f),
        npu_float_to_bits(5.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC0_PORT);
    npu_spm_clear_slot(SRC1_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC0_PORT, lhs, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC1_PORT, rhs, ELEM_COUNT);
    npu_golden_vpu_elemwise_f32(VPU_OP_VMUL, lhs, rhs, expected, ELEM_COUNT);

    vpu_cmd_launch_binary(VPU_DEVICE_ID, VPU_OP_VMUL, SYNC_INDICATOR, 0x3U,
                          0x4U, 1U, ELEM_COUNT, sizeof(uint32_t),
                          sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_float_vector_close(
            NULL, npu_spm_slot_word_ptr_default(DST_PORT), expected,
            ELEM_COUNT, 60000000ULL, 0.03f, 0.03f) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_ELEMWISE_MUL_F32_FAIL\n");
        npu_expect_float_vector_close("VPU_ELEMWISE_MUL_F32", expected,
                                      actual, ELEM_COUNT, 0.03f, 0.03f);
        return 1;
    }

    printf("VPU_ELEMWISE_MUL_F32_PASS\n");
    return 0;
}
