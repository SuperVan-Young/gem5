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
#include "softmax.hh"
#include "swiglu.hh"

enum LlmPrimitiveLayout
{
    VPU_DEVICE_ID = 0U,
    ROWS = 2U,
    COLS = 8U,
    RMS_SRC_SLOT = 0U,
    RMS_WEIGHT_SLOT = 1U,
    RMS_DST_SLOT = 2U,
    RMS_SCRATCH_BASE = 3U,
    RMS_SYNC = 0x81U,
    SOFTMAX_SRC_SLOT = 10U,
    SOFTMAX_DST_SLOT = 11U,
    SOFTMAX_SCRATCH_BASE = 12U,
    SOFTMAX_SYNC = 0x82U,
    SWIGLU_GATE_SLOT = 18U,
    SWIGLU_VALUE_SLOT = 19U,
    SWIGLU_DST_SLOT = 20U,
    SWIGLU_SCRATCH_BASE = 21U,
    SWIGLU_SYNC = 0x83U,
};

int
main(void)
{
    const uint32_t rms_src[ROWS * COLS] = {
        npu_float_to_bits(0.5f),  npu_float_to_bits(-1.0f),
        npu_float_to_bits(2.0f),  npu_float_to_bits(-0.25f),
        npu_float_to_bits(0.75f), npu_float_to_bits(1.25f),
        npu_float_to_bits(-1.5f), npu_float_to_bits(0.625f),
        npu_float_to_bits(1.0f),  npu_float_to_bits(-0.75f),
        npu_float_to_bits(0.25f), npu_float_to_bits(1.5f),
        npu_float_to_bits(-1.25f), npu_float_to_bits(0.875f),
        npu_float_to_bits(0.125f), npu_float_to_bits(-0.5f),
    };
    const uint32_t rms_weight[COLS] = {
        npu_float_to_bits(1.0f),  npu_float_to_bits(0.9f),
        npu_float_to_bits(1.1f),  npu_float_to_bits(1.2f),
        npu_float_to_bits(0.8f),  npu_float_to_bits(1.05f),
        npu_float_to_bits(0.95f), npu_float_to_bits(1.15f),
    };
    const uint32_t softmax_src[ROWS * COLS] = {
        npu_float_to_bits(-2.0f), npu_float_to_bits(-1.0f),
        npu_float_to_bits(0.0f),  npu_float_to_bits(1.0f),
        npu_float_to_bits(2.0f),  npu_float_to_bits(0.5f),
        npu_float_to_bits(-0.5f), npu_float_to_bits(1.5f),
        npu_float_to_bits(0.25f), npu_float_to_bits(0.75f),
        npu_float_to_bits(-1.25f), npu_float_to_bits(2.25f),
        npu_float_to_bits(-0.75f), npu_float_to_bits(1.25f),
        npu_float_to_bits(1.75f), npu_float_to_bits(-0.25f),
    };
    const uint32_t swiglu_gate[ROWS * COLS] = {
        npu_float_to_bits(-1.5f), npu_float_to_bits(-0.5f),
        npu_float_to_bits(0.25f), npu_float_to_bits(1.0f),
        npu_float_to_bits(1.5f),  npu_float_to_bits(-2.0f),
        npu_float_to_bits(0.75f), npu_float_to_bits(-0.25f),
        npu_float_to_bits(0.1f),  npu_float_to_bits(-0.2f),
        npu_float_to_bits(0.3f),  npu_float_to_bits(-0.4f),
        npu_float_to_bits(0.5f),  npu_float_to_bits(-0.6f),
        npu_float_to_bits(0.7f),  npu_float_to_bits(-0.8f),
    };
    const uint32_t swiglu_value[ROWS * COLS] = {
        npu_float_to_bits(0.75f), npu_float_to_bits(1.5f),
        npu_float_to_bits(-0.5f), npu_float_to_bits(2.0f),
        npu_float_to_bits(-1.0f), npu_float_to_bits(0.25f),
        npu_float_to_bits(1.25f), npu_float_to_bits(-0.75f),
        npu_float_to_bits(1.0f),  npu_float_to_bits(-1.0f),
        npu_float_to_bits(0.5f),  npu_float_to_bits(-0.5f),
        npu_float_to_bits(1.5f),  npu_float_to_bits(-1.5f),
        npu_float_to_bits(2.0f),  npu_float_to_bits(-2.0f),
    };
    uint32_t rms_expected[ROWS * COLS];
    uint32_t softmax_expected[ROWS * COLS];
    uint32_t swiglu_expected[ROWS * COLS];
    uint32_t actual[ROWS * COLS];
    const float rms_eps = 0.125f;

    for (uint32_t slot = 0U; slot < 27U; ++slot) {
        npu_spm_clear_slot(slot);
    }

    llm_store_logical_matrix_last_axis_front(RMS_SRC_SLOT, rms_src, ROWS, COLS);
    npu_spm_store_u32_vector(RMS_WEIGHT_SLOT, rms_weight, COLS);
    llm_store_logical_matrix_last_axis_front(
        SOFTMAX_SRC_SLOT, softmax_src, ROWS, COLS);
    llm_store_logical_matrix_last_axis_front(
        SWIGLU_GATE_SLOT, swiglu_gate, ROWS, COLS);
    llm_store_logical_matrix_last_axis_front(
        SWIGLU_VALUE_SLOT, swiglu_value, ROWS, COLS);

    npu_golden_rmsnorm_lastdim_f32(
        rms_src, rms_weight, ROWS, COLS, rms_eps, rms_expected);
    npu_golden_softmax_lastdim_f32(
        softmax_src, ROWS, COLS, softmax_expected);
    npu_golden_swiglu_lastdim_f32(
        swiglu_gate, swiglu_value, ROWS, COLS, swiglu_expected);

    const PrimitiveTensorDesc rms_src_desc =
        llm_packed_last_axis_tensor(RMS_SRC_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc rms_weight_desc =
        PrimitiveTensorDesc::denseSpm(RMS_WEIGHT_SLOT, VPU_DATA_F32, {COLS});
    const PrimitiveTensorDesc rms_dst_desc =
        llm_packed_last_axis_tensor(RMS_DST_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc softmax_src_desc =
        llm_packed_last_axis_tensor(SOFTMAX_SRC_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc softmax_dst_desc =
        llm_packed_last_axis_tensor(SOFTMAX_DST_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc swiglu_gate_desc =
        llm_packed_last_axis_tensor(SWIGLU_GATE_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc swiglu_value_desc =
        llm_packed_last_axis_tensor(SWIGLU_VALUE_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc swiglu_dst_desc =
        llm_packed_last_axis_tensor(SWIGLU_DST_SLOT, ROWS, COLS);

    const size_t rms_count = vpu_primitive_rmsnorm_f32(
        VPU_DEVICE_ID, rms_src_desc, rms_weight_desc, rms_dst_desc,
        RMS_SCRATCH_BASE, 0U, rms_eps, RMS_SYNC);
    npu_launch_sync_wait(VPU_DEVICE_ID, RMS_SYNC, 0U, 0U, 0U);
    const size_t softmax_count = vpu_primitive_softmax_f32(
        VPU_DEVICE_ID, softmax_src_desc, softmax_dst_desc,
        SOFTMAX_SCRATCH_BASE, 16U, SOFTMAX_SYNC);
    npu_launch_sync_wait(VPU_DEVICE_ID, SOFTMAX_SYNC, 0U, 0U, 0U);
    const size_t swiglu_count = vpu_primitive_swiglu_f32(
        VPU_DEVICE_ID, swiglu_gate_desc, swiglu_value_desc, swiglu_dst_desc,
        SWIGLU_SCRATCH_BASE, 26U, SWIGLU_SYNC);
    npu_launch_sync_wait(VPU_DEVICE_ID, SWIGLU_SYNC, 0U, 0U, 0U);
    npu_cmd_sync_done();

    llm_load_logical_matrix_last_axis_front(RMS_DST_SLOT, actual, ROWS, COLS);
    if (npu_expect_float_vector_close(
            "LLM_PRIMITIVES_RMSNORM", rms_expected, actual, ROWS * COLS, 0.01f,
            0.01f) != 0) {
        printf("LLM_PRIMITIVES_F32_FAIL\n");
        return 1;
    }
    llm_load_logical_matrix_last_axis_front(
        SOFTMAX_DST_SLOT, actual, ROWS, COLS);
    if (npu_expect_float_vector_close(
            "LLM_PRIMITIVES_SOFTMAX", softmax_expected, actual, ROWS * COLS,
            0.03f, 0.03f) != 0) {
        printf("LLM_PRIMITIVES_F32_FAIL\n");
        return 1;
    }
    llm_load_logical_matrix_last_axis_front(
        SWIGLU_DST_SLOT, actual, ROWS, COLS);
    if (npu_expect_float_vector_close(
            "LLM_PRIMITIVES_SWIGLU", swiglu_expected, actual, ROWS * COLS,
            0.03f, 0.03f) != 0) {
        printf("LLM_PRIMITIVES_F32_FAIL\n");
        return 1;
    }

    printf("LLM_PRIMITIVES_RMSNORM_CACHED_CMDS=%u\n",
           (unsigned)rms_count);
    printf("LLM_PRIMITIVES_SOFTMAX_CACHED_CMDS=%u\n",
           (unsigned)softmax_count);
    printf("LLM_PRIMITIVES_SWIGLU_CACHED_CMDS=%u\n",
           (unsigned)swiglu_count);
    printf("LLM_PRIMITIVES_F32_PASS\n");
    return 0;
}
