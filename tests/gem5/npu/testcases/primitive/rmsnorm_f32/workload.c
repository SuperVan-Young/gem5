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
#include "rmsnorm.hh"

enum RmsNormLayout
{
    VPU_DEVICE_ID = 0U,
    ROWS = 2U,
    COLS = 8U,
    SRC_SLOT = 0U,
    WEIGHT_SLOT = 1U,
    DST_SLOT = 2U,
    SCRATCH_BASE = 3U,
    SYNC_INDICATOR = 0x51U,
};

int
main(void)
{
    const uint32_t src[ROWS * COLS] = {
        npu_float_to_bits(0.5f),   npu_float_to_bits(-1.0f),
        npu_float_to_bits(2.0f),   npu_float_to_bits(-0.25f),
        npu_float_to_bits(0.75f),  npu_float_to_bits(1.25f),
        npu_float_to_bits(-1.5f),  npu_float_to_bits(0.625f),
        npu_float_to_bits(1.0f),   npu_float_to_bits(-0.75f),
        npu_float_to_bits(0.25f),  npu_float_to_bits(1.5f),
        npu_float_to_bits(-1.25f), npu_float_to_bits(0.875f),
        npu_float_to_bits(0.125f), npu_float_to_bits(-0.5f),
    };
    const uint32_t weight[COLS] = {
        npu_float_to_bits(1.0f),  npu_float_to_bits(0.9f),
        npu_float_to_bits(1.1f),  npu_float_to_bits(1.2f),
        npu_float_to_bits(0.8f),  npu_float_to_bits(1.05f),
        npu_float_to_bits(0.95f), npu_float_to_bits(1.15f),
    };
    uint32_t expected[ROWS * COLS];
    uint32_t actual[ROWS * COLS];
    const float epsilon = 0.125f;

    for (uint32_t slot = 0U; slot < 10U; ++slot) {
        npu_spm_clear_slot(slot);
    }

    llm_store_logical_matrix_last_axis_front(SRC_SLOT, src, ROWS, COLS);
    npu_spm_store_u32_vector(WEIGHT_SLOT, weight, COLS);
    npu_golden_rmsnorm_lastdim_f32(src, weight, ROWS, COLS, epsilon, expected);

    const size_t macro_count = vpu_primitive_rmsnorm_f32(
        VPU_DEVICE_ID, llm_packed_last_axis_tensor(SRC_SLOT, ROWS, COLS),
        PrimitiveTensorDesc::denseSpm(WEIGHT_SLOT, VPU_DATA_F32, {COLS}),
        llm_packed_last_axis_tensor(DST_SLOT, ROWS, COLS), SCRATCH_BASE, 0U,
        epsilon, SYNC_INDICATOR);
    npu_launch_sync_wait(VPU_DEVICE_ID, SYNC_INDICATOR, 0U, 0U, 0U);
    npu_cmd_sync_done();

    llm_load_logical_matrix_last_axis_front(DST_SLOT, actual, ROWS, COLS);
    if (npu_expect_float_vector_close("RMSNORM_F32", expected, actual,
                                      ROWS * COLS, 0.01f, 0.01f) != 0) {
        printf("RMSNORM_F32_FAIL\n");
        return 1;
    }

    printf("RMSNORM_F32_CACHED_CMDS=%u\n", (unsigned)macro_count);
    printf("RMSNORM_F32_PASS\n");
    return 0;
}
