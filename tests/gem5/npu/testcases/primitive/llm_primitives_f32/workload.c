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
#include "rmsnorm.hh"
#include "softmax.hh"
#include "swiglu.hh"

enum
{
    VPU_DEVICE_ID = 0U,
    ROWS = 128U,
    COLS = 128U,
    MAT_SLOT_SPAN = ((ROWS * COLS * sizeof(uint32_t)) +
                     VPU_LOCAL_SLOT_STRIDE - 1U) / VPU_LOCAL_SLOT_STRIDE,
    VEC_SLOT_SPAN = ((COLS * sizeof(uint32_t)) +
                     VPU_LOCAL_SLOT_STRIDE - 1U) / VPU_LOCAL_SLOT_STRIDE,
    RMS_SRC_SLOT = 0U,
    RMS_WEIGHT_SLOT = RMS_SRC_SLOT + MAT_SLOT_SPAN,
    RMS_DST_SLOT = RMS_WEIGHT_SLOT + VEC_SLOT_SPAN,
    RMS_SCRATCH_BASE = RMS_DST_SLOT + MAT_SLOT_SPAN,
    RMS_SYNC = 0x81U,
    SOFTMAX_SRC_SLOT = RMS_SCRATCH_BASE + MAT_SLOT_SPAN * 8U,
    SOFTMAX_DST_SLOT = SOFTMAX_SRC_SLOT + MAT_SLOT_SPAN,
    SOFTMAX_SCRATCH_BASE = SOFTMAX_DST_SLOT + MAT_SLOT_SPAN,
    SOFTMAX_SYNC = 0x82U,
    SWIGLU_GATE_SLOT = SOFTMAX_SCRATCH_BASE + MAT_SLOT_SPAN * 5U,
    SWIGLU_VALUE_SLOT = SWIGLU_GATE_SLOT + MAT_SLOT_SPAN,
    SWIGLU_DST_SLOT = SWIGLU_VALUE_SLOT + MAT_SLOT_SPAN,
    SWIGLU_SCRATCH_BASE = SWIGLU_DST_SLOT + MAT_SLOT_SPAN,
    SWIGLU_SYNC = 0x83U,
};

int
main(void)
{
    std::vector<uint32_t> rms_src(ROWS * COLS);
    std::vector<uint32_t> rms_weight(COLS);
    std::vector<uint32_t> softmax_src(ROWS * COLS);
    std::vector<uint32_t> swiglu_gate(ROWS * COLS);
    std::vector<uint32_t> swiglu_value(ROWS * COLS);
    std::vector<uint32_t> rms_expected(ROWS * COLS);
    std::vector<uint32_t> softmax_expected(ROWS * COLS);
    std::vector<uint32_t> swiglu_expected(ROWS * COLS);
    std::vector<uint32_t> actual(ROWS * COLS);
    const float rms_eps = 0.125f;

    for (uint32_t row = 0U; row < ROWS; ++row) {
        for (uint32_t col = 0U; col < COLS; ++col) {
            const uint32_t index = row * COLS + col;
            rms_src[index] = npu_float_to_bits(
                (static_cast<int32_t>(row % 23U) - 11) * 0.0625f +
                (static_cast<int32_t>(col % 29U) - 14) * 0.015625f);
            softmax_src[index] = npu_float_to_bits(
                (static_cast<int32_t>(row % 31U) - 15) * 0.05f +
                (static_cast<int32_t>(col % 27U) - 13) * 0.07f);
            swiglu_gate[index] = npu_float_to_bits(
                (static_cast<int32_t>(row % 21U) - 10) * 0.11f +
                (static_cast<int32_t>(col % 17U) - 8) * 0.07f);
            swiglu_value[index] = npu_float_to_bits(
                (static_cast<int32_t>(row % 15U) - 7) * 0.09f -
                (static_cast<int32_t>(col % 19U) - 9) * 0.05f);
        }
    }
    for (uint32_t col = 0U; col < COLS; ++col) {
        rms_weight[col] =
            npu_float_to_bits(0.75f + static_cast<float>(col % 37U) * 0.01f);
    }

    llm_clear_slot_span(0U, SWIGLU_SCRATCH_BASE + MAT_SLOT_SPAN * 5U);

    llm_store_logical_matrix_last_axis_front(RMS_SRC_SLOT, rms_src.data(), ROWS,
                                             COLS);
    npu_spm_store_u32_vector(RMS_WEIGHT_SLOT, rms_weight.data(), COLS);
    llm_store_logical_matrix_last_axis_front(
        SOFTMAX_SRC_SLOT, softmax_src.data(), ROWS, COLS);
    llm_store_logical_matrix_last_axis_front(
        SWIGLU_GATE_SLOT, swiglu_gate.data(), ROWS, COLS);
    llm_store_logical_matrix_last_axis_front(
        SWIGLU_VALUE_SLOT, swiglu_value.data(), ROWS, COLS);

    npu_golden_rmsnorm_lastdim_f32(
        rms_src.data(), rms_weight.data(), ROWS, COLS, rms_eps,
        rms_expected.data());
    npu_golden_softmax_lastdim_f32(
        softmax_src.data(), ROWS, COLS, softmax_expected.data());
    npu_golden_swiglu_lastdim_f32(
        swiglu_gate.data(), swiglu_value.data(), ROWS, COLS,
        swiglu_expected.data());

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

    llm_load_logical_matrix_last_axis_front(RMS_DST_SLOT, actual.data(), ROWS,
                                            COLS);
    if (npu_expect_float_vector_close(
            "LLM_PRIMITIVES_RMSNORM", rms_expected.data(), actual.data(),
            ROWS * COLS, 0.01f, 0.01f) != 0) {
        printf("LLM_PRIMITIVES_F32_FAIL\n");
        return 1;
    }
    llm_load_logical_matrix_last_axis_front(
        SOFTMAX_DST_SLOT, actual.data(), ROWS, COLS);
    if (npu_expect_float_vector_close(
            "LLM_PRIMITIVES_SOFTMAX", softmax_expected.data(), actual.data(),
            ROWS * COLS, 0.03f, 0.03f) != 0) {
        printf("LLM_PRIMITIVES_F32_FAIL\n");
        return 1;
    }
    llm_load_logical_matrix_last_axis_front(
        SWIGLU_DST_SLOT, actual.data(), ROWS, COLS);
    if (npu_expect_float_vector_close(
            "LLM_PRIMITIVES_SWIGLU", swiglu_expected.data(), actual.data(),
            ROWS * COLS, 0.03f, 0.03f) != 0) {
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
