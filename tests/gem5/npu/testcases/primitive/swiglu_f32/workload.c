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
#include "swiglu.hh"

enum
{
    VPU_DEVICE_ID = 0U,
    ROWS = 128U,
    COLS = 128U,
    MAT_SLOT_SPAN = ((ROWS * COLS * sizeof(uint32_t)) +
                     VPU_LOCAL_SLOT_STRIDE - 1U) / VPU_LOCAL_SLOT_STRIDE,
    GATE_SLOT = 0U,
    VALUE_SLOT = GATE_SLOT + MAT_SLOT_SPAN,
    DST_SLOT = VALUE_SLOT + MAT_SLOT_SPAN,
    SCRATCH_BASE = DST_SLOT + MAT_SLOT_SPAN,
    SYNC_INDICATOR = 0x71U,
};

int
main(void)
{
    std::vector<uint32_t> gate(ROWS * COLS);
    std::vector<uint32_t> value(ROWS * COLS);
    std::vector<uint32_t> expected(ROWS * COLS);
    std::vector<uint32_t> actual(ROWS * COLS);

    for (uint32_t row = 0U; row < ROWS; ++row) {
        for (uint32_t col = 0U; col < COLS; ++col) {
            const uint32_t index = row * COLS + col;
            const float gate_value =
                (static_cast<int32_t>(row % 21U) - 10) * 0.11f +
                (static_cast<int32_t>(col % 17U) - 8) * 0.07f;
            const float value_value =
                (static_cast<int32_t>(row % 15U) - 7) * 0.09f -
                (static_cast<int32_t>(col % 19U) - 9) * 0.05f;
            gate[index] = npu_float_to_bits(gate_value);
            value[index] = npu_float_to_bits(value_value);
        }
    }

    llm_clear_slot_span(GATE_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(VALUE_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(DST_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(SCRATCH_BASE, MAT_SLOT_SPAN * 5U + 8U);

    llm_store_logical_matrix_last_axis_front(GATE_SLOT, gate.data(), ROWS, COLS);
    llm_store_logical_matrix_last_axis_front(VALUE_SLOT, value.data(), ROWS,
                                             COLS);
    npu_golden_swiglu_lastdim_f32(gate.data(), value.data(), ROWS, COLS,
                                  expected.data());

    const size_t macro_count = vpu_primitive_swiglu_f32(
        VPU_DEVICE_ID, llm_packed_last_axis_tensor(GATE_SLOT, ROWS, COLS),
        llm_packed_last_axis_tensor(VALUE_SLOT, ROWS, COLS),
        llm_packed_last_axis_tensor(DST_SLOT, ROWS, COLS), SCRATCH_BASE, 0U,
        SYNC_INDICATOR);
    npu_launch_sync_wait(VPU_DEVICE_ID, SYNC_INDICATOR, 0U, 0U, 0U);
    npu_cmd_sync_done();

    llm_load_logical_matrix_last_axis_front(DST_SLOT, actual.data(), ROWS,
                                            COLS);
    if (npu_expect_float_vector_close("SWIGLU_F32", expected.data(),
                                      actual.data(),
                                      ROWS * COLS, 0.03f, 0.03f) != 0) {
        printf("SWIGLU_F32_FAIL\n");
        return 1;
    }

    printf("SWIGLU_F32_CACHED_CMDS=%u\n", (unsigned)macro_count);
    printf("SWIGLU_F32_PASS\n");
    return 0;
}
