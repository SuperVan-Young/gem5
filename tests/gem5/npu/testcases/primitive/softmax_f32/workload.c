/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "golden/llm_f32.hh"
#include "llm_test_utils.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "softmax.hh"

enum SoftmaxLayout
{
    VPU_DEVICE_ID = 0U,
    ROWS = 2U,
    COLS = 8U,
    SRC_SLOT = 0U,
    DST_SLOT = 1U,
    SCRATCH_BASE = 2U,
    SYNC_INDICATOR = 0x61U,
};

int
main(void)
{
    const uint32_t src[ROWS * COLS] = {
        npu_float_to_bits(-2.0f), npu_float_to_bits(-1.0f),
        npu_float_to_bits(0.0f),  npu_float_to_bits(1.0f),
        npu_float_to_bits(2.0f),  npu_float_to_bits(0.5f),
        npu_float_to_bits(-0.5f), npu_float_to_bits(1.5f),
        npu_float_to_bits(0.25f), npu_float_to_bits(0.75f),
        npu_float_to_bits(-1.25f), npu_float_to_bits(2.25f),
        npu_float_to_bits(-0.75f), npu_float_to_bits(1.25f),
        npu_float_to_bits(1.75f), npu_float_to_bits(-0.25f),
    };
    uint32_t expected[ROWS * COLS];
    uint32_t actual[ROWS * COLS];

    for (uint32_t slot = 0U; slot < 6U; ++slot) {
        npu_spm_clear_slot(slot);
    }

    llm_store_logical_matrix_last_axis_front(SRC_SLOT, src, ROWS, COLS);
    npu_golden_softmax_lastdim_f32(src, ROWS, COLS, expected);

    const size_t macro_count = vpu_primitive_softmax_f32(
        VPU_DEVICE_ID, llm_packed_last_axis_tensor(SRC_SLOT, ROWS, COLS),
        llm_packed_last_axis_tensor(DST_SLOT, ROWS, COLS), SCRATCH_BASE, 0U,
        SYNC_INDICATOR);
    npu_launch_sync_wait(VPU_DEVICE_ID, SYNC_INDICATOR, 0U, 0U, 0U);
    npu_cmd_sync_done();

    llm_load_logical_matrix_last_axis_front(DST_SLOT, actual, ROWS, COLS);
    if (npu_expect_float_vector_close("SOFTMAX_F32", expected, actual,
                                      ROWS * COLS, 0.03f, 0.03f) != 0) {
        printf("SOFTMAX_F32_FAIL\n");
        return 1;
    }

    printf("SOFTMAX_F32_CACHED_CMDS=%u\n", (unsigned)macro_count);
    printf("SOFTMAX_F32_PASS\n");
    return 0;
}
