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
    SLICES = 3U,
    ROWS = 64U,
    COLS = 128U,
    PER_SLICE_SLOT_SPAN =
        ((ROWS * COLS * sizeof(uint32_t)) + VPU_LOCAL_SLOT_STRIDE - 1U) /
        VPU_LOCAL_SLOT_STRIDE,
    REDUCE_SLOT_SPAN =
        ((ROWS * sizeof(uint32_t)) + VPU_LOCAL_SLOT_STRIDE - 1U) /
        VPU_LOCAL_SLOT_STRIDE,
    MAT_SLOT_SPAN = SLICES * PER_SLICE_SLOT_SPAN,
    SCRATCH_SLICE_SLOT_SPAN =
        REDUCE_SLOT_SPAN + PER_SLICE_SLOT_SPAN + PER_SLICE_SLOT_SPAN +
        REDUCE_SLOT_SPAN,
    SCRATCH_SLOT_SPAN = SLICES * SCRATCH_SLICE_SLOT_SPAN,
    SRC_SLOT = 0U,
    DST_SLOT = SRC_SLOT + MAT_SLOT_SPAN,
    SCRATCH_BASE = DST_SLOT + MAT_SLOT_SPAN,
    SYNC_INDICATOR = 0x62U,
};

int
main(void)
{
    std::vector<uint32_t> src(SLICES * ROWS * COLS);
    std::vector<uint32_t> expected(SLICES * ROWS * COLS);
    std::vector<uint32_t> actual(SLICES * ROWS * COLS);

    for (uint32_t slice = 0U; slice < SLICES; ++slice) {
        for (uint32_t row = 0U; row < ROWS; ++row) {
            for (uint32_t col = 0U; col < COLS; ++col) {
                const uint32_t index =
                    (slice * ROWS * COLS) + (row * COLS) + col;
                const float value =
                    (static_cast<int32_t>(slice) - 1) * 0.11f +
                    (static_cast<int32_t>(row % 17U) - 8) * 0.06f +
                    (static_cast<int32_t>(col % 29U) - 14) * 0.05f;
                src[index] = npu_float_to_bits(value);
            }
        }
    }

    llm_clear_slot_span(SRC_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(DST_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(SCRATCH_BASE, SCRATCH_SLOT_SPAN);

    llm_store_logical_tensor3_last_axis_front(
        SRC_SLOT, src.data(), SLICES, ROWS, COLS);
    for (uint32_t slice = 0U; slice < SLICES; ++slice) {
        npu_golden_softmax_lastdim_f32(
            src.data() + (slice * ROWS * COLS), ROWS, COLS,
            expected.data() + (slice * ROWS * COLS));
    }

    PrimitiveSyncDesc sync = {};
    sync.syncIndicator = SYNC_INDICATOR;
    sync.setSnsIndicator = true;

    const size_t macro_count = vpu_primitive_softmax_f32(
        VPU_DEVICE_ID,
        llm_packed_last_axis_tensor_3d(SRC_SLOT, SLICES, ROWS, COLS),
        llm_packed_last_axis_tensor_3d(DST_SLOT, SLICES, ROWS, COLS),
        SCRATCH_BASE, 0U, sync);
    npu_launch_sync_wait(VPU_DEVICE_ID, SYNC_INDICATOR, 0U, 0U, 0U);
    npu_cmd_sync_done();

    llm_load_logical_tensor3_last_axis_front(
        DST_SLOT, actual.data(), SLICES, ROWS, COLS);
    if (npu_expect_float_vector_close("SOFTMAX_3X64X128XF32", expected.data(),
                                      actual.data(), SLICES * ROWS * COLS,
                                      0.03f, 0.03f) != 0) {
        printf("SOFTMAX_3X64X128XF32_FAIL\n");
        return 1;
    }

    printf("SOFTMAX_3X64X128XF32_CACHED_CMDS=%u\n", (unsigned)macro_count);
    printf("SOFTMAX_3X64X128XF32_PASS\n");
    return 0;
}
