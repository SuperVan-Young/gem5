/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "golden/llm_f32.hh"
#include "llm_test_utils.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "softmax.hh"

enum
{
    VPU_DEVICE_ID = 0U,
    ROWS = 128U,
    COLS = 128U,
    MAT_SLOT_SPAN = ((ROWS * COLS * sizeof(uint32_t)) +
                     VPU_LOCAL_SLOT_STRIDE - 1U) / VPU_LOCAL_SLOT_STRIDE,
    SRC_SLOT = 0U,
    DST_SLOT = SRC_SLOT + MAT_SLOT_SPAN,
    SCRATCH_BASE = DST_SLOT + MAT_SLOT_SPAN,
    SYNC_INDICATOR = 0x61U,
};

int
main(void)
{
    std::vector<uint32_t> src(ROWS * COLS);
    std::vector<uint32_t> expected(ROWS * COLS);
    std::vector<uint32_t> actual(ROWS * COLS);

    for (uint32_t row = 0U; row < ROWS; ++row) {
        for (uint32_t col = 0U; col < COLS; ++col) {
            const uint32_t index = row * COLS + col;
            const float value =
                (static_cast<int32_t>(row % 31U) - 15) * 0.05f +
                (static_cast<int32_t>(col % 27U) - 13) * 0.07f;
            src[index] = npu_float_to_bits(value);
        }
    }

    llm_clear_slot_span(SRC_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(DST_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(SCRATCH_BASE, MAT_SLOT_SPAN * 4U + 16U);

    llm_store_logical_matrix_last_axis_front(SRC_SLOT, src.data(), ROWS, COLS);
    npu_golden_softmax_lastdim_f32(src.data(), ROWS, COLS, expected.data());

    const size_t macro_count = vpu_primitive_softmax_f32(
        VPU_DEVICE_ID, llm_packed_last_axis_tensor(SRC_SLOT, ROWS, COLS),
        llm_packed_last_axis_tensor(DST_SLOT, ROWS, COLS), SCRATCH_BASE, 0U,
        SYNC_INDICATOR);
    npu_launch_sync_wait(VPU_DEVICE_ID, SYNC_INDICATOR, 0U, 0U, 0U);
    npu_cmd_sync_done();

    llm_load_logical_matrix_last_axis_front(DST_SLOT, actual.data(), ROWS,
                                            COLS);
    if (npu_expect_float_vector_close("SOFTMAX_F32", expected.data(),
                                      actual.data(),
                                      ROWS * COLS, 0.03f, 0.03f) != 0) {
        printf("SOFTMAX_F32_FAIL\n");
        return 1;
    }

    printf("SOFTMAX_F32_CACHED_CMDS=%u\n", (unsigned)macro_count);
    printf("SOFTMAX_F32_PASS\n");
    return 0;
}
