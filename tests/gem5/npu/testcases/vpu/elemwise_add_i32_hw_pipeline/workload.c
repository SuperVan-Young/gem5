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

enum ElemwiseAddI32HwPipelineLayout
{
    VPU_DEVICE_ID = 0U,
    SRC0_PORT = 0U,
    SRC1_PORT = 1U,
    DST_PORT = 2U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x91U,
};

int
main(void)
{
    const uint32_t lhs[ELEM_COUNT] = {10U, 20U, 30U, 40U};
    const uint32_t rhs[ELEM_COUNT] = {1U, 2U, 3U, 4U};
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];
    npu_spm_clear_slot(SRC0_PORT);
    npu_spm_clear_slot(SRC1_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC0_PORT, lhs, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC1_PORT, rhs, ELEM_COUNT);
    npu_golden_vpu_elemwise_i32(VPU_OP_VADD, lhs, rhs, expected, ELEM_COUNT);

    vpu_cmd_launch_binary(VPU_DEVICE_ID, VPU_OP_VADD, SYNC_INDICATOR, 0x3U,
                          0x4U, 1U, ELEM_COUNT, sizeof(uint32_t),
                          sizeof(uint32_t), VPU_DATA_I32);

    if (npu_wait_u32_vector_match(NULL,
                                  npu_spm_slot_word_ptr_default(DST_PORT),
                                  expected, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_ELEMWISE_ADD_I32_HW_PIPELINE_FAIL\n");
        npu_expect_u32_vector("VPU_ELEMWISE_ADD_I32_HW_PIPELINE", expected,
                              actual, ELEM_COUNT);
        return 1;
    }

    printf("VPU_ELEMWISE_ADD_I32_HW_PIPELINE_PASS\n");
    return 0;
}
