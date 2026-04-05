/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum DecodeVaddSmokeLayout
{
    VPU_DEVICE_ID = 0U,
    SRC0_PORT = 1U,
    SRC1_PORT = 2U,
    DST_PORT = 3U,
    ELEM_COUNT = 4U,
    VADD_SYNC = 0x32U,
};

int
main(void)
{
    const uint32_t src0[ELEM_COUNT] = {1U, 3U, 5U, 7U};
    const uint32_t src1[ELEM_COUNT] = {2U, 4U, 6U, 8U};
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC0_PORT);
    npu_spm_clear_slot(SRC1_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC0_PORT, src0, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC1_PORT, src1, ELEM_COUNT);

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected[idx] = src0[idx] + src1[idx];
    }

    vpu_cmd_launch_binary(VPU_DEVICE_ID, VPU_OP_VADD, VADD_SYNC, 0x6U, 0x8U,
                          1U, ELEM_COUNT, sizeof(uint32_t),
                          sizeof(uint32_t), VPU_DATA_I32);

    if (npu_wait_u32_vector_match(
            NULL, npu_spm_slot_word_ptr_default(DST_PORT), expected,
            ELEM_COUNT, 60000000ULL) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_DECODE_VADD_SMOKE_FAIL\n");
        npu_expect_u32_vector("VPU_DECODE_VADD_SMOKE", expected, actual,
                              ELEM_COUNT);
        return 1;
    }

    printf("VPU_DECODE_VADD_SMOKE_PASS\n");
    return 0;
}
