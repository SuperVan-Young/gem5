/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "golden/vpu_softmax.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum SoftmaxRampLayout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x81U,
};

int
main(void)
{
    const uint32_t src_bits[ELEM_COUNT] = {
        npu_float_to_bits(1.0f),
        npu_float_to_bits(2.0f),
        npu_float_to_bits(3.0f),
        npu_float_to_bits(4.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_softmax_f32(src_bits, expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VSOFTMAX, SYNC_INDICATOR,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_float_vector_close(NULL,
                                    npu_spm_slot_word_ptr_default(DST_PORT),
                                    expected, ELEM_COUNT, 80000000ULL, 0.03f,
                                    0.03f) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_SOFTMAX_RAMP_FAIL\n");
        npu_expect_float_vector_close("VPU_SOFTMAX_RAMP", expected, actual,
                                      ELEM_COUNT, 0.03f, 0.03f);
        return 1;
    }

    printf("VPU_SOFTMAX_RAMP_PASS\n");
    return 0;
}

