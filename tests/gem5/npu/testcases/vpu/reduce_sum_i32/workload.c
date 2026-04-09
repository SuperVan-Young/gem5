/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "golden/vpu_reduce.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum ReduceSumI32Layout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x71U,
};

int
main(void)
{
    const int32_t src_i32[ELEM_COUNT] = {2, -3, 4, 7};
    uint32_t src_bits[ELEM_COUNT];
    int32_t expected_i32 = 0;
    uint32_t expected = 0;

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src_bits[idx], &src_i32[idx], sizeof(src_bits[idx]));
    }

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_reduce_sum_i32(src_i32, &expected_i32, ELEM_COUNT);
    memcpy(&expected, &expected_i32, sizeof(expected));

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VREDUCE_SUM, SYNC_INDICATOR,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_I32);

    const volatile uint32_t *expected_slot =
        npu_spm_slot_word_ptr_default(DST_PORT);

    if (npu_wait_u32_scalar_match(NULL, expected_slot, expected,
                                  60000000ULL) != 0) {
        printf("VPU_REDUCE_SUM_I32_FAIL exp=%d act=%d\n", expected_i32,
               (int32_t)npu_spm_slot_word_ptr_default(DST_PORT)[0]);
        return 1;
    }

    printf("VPU_REDUCE_SUM_I32_PASS\n");
    return 0;
}
