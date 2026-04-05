/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "golden/vpu_reduce.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum ReduceMaxF32Layout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x73U,
};

int
main(void)
{
    const uint32_t src_bits[ELEM_COUNT] = {
        npu_float_to_bits(-3.0f),
        npu_float_to_bits(5.5f),
        npu_float_to_bits(2.25f),
        npu_float_to_bits(4.0f),
    };
    uint32_t expected = 0U;

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_reduce_max_f32(src_bits, &expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VREDUCE_MAX, SYNC_INDICATOR,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    const volatile uint32_t *expected_slot =
        npu_spm_slot_word_ptr_default(DST_PORT);

    if (npu_wait_u32_scalar_match(NULL, expected_slot, expected,
                                  60000000ULL) != 0) {
        printf("VPU_REDUCE_MAX_F32_FAIL exp=%#x act=%#x\n", expected,
               npu_spm_slot_word_ptr_default(DST_PORT)[0]);
        return 1;
    }

    printf("VPU_REDUCE_MAX_F32_PASS\n");
    return 0;
}
