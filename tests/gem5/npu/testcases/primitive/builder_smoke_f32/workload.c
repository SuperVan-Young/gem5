/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "llm_test_utils.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "binary.hh"
#include "unary.hh"
#include "vpu_elemwise.hh"
#include "vpu_unary.hh"

enum
{
    VPU_DEVICE_ID = 0U,
    ROWS = 128U,
    COLS = 128U,
    MAT_SLOT_SPAN = ((ROWS * COLS * sizeof(uint32_t)) +
                     VPU_LOCAL_SLOT_STRIDE - 1U) / VPU_LOCAL_SLOT_STRIDE,
    SRC_SLOT = 0U,
    RHS_SLOT = SRC_SLOT + MAT_SLOT_SPAN,
    EXP_DST_SLOT = RHS_SLOT + MAT_SLOT_SPAN,
    ADD_DST_SLOT = EXP_DST_SLOT + MAT_SLOT_SPAN,
    EXP_SYNC = 0x41U,
    ADD_SYNC = 0x42U,
};

int
main(void)
{
    std::vector<uint32_t> src(ROWS * COLS);
    std::vector<uint32_t> rhs(ROWS * COLS);
    std::vector<uint32_t> exp_expected(ROWS * COLS);
    std::vector<uint32_t> add_expected(ROWS * COLS);
    std::vector<uint32_t> actual(ROWS * COLS);

    for (uint32_t row = 0U; row < ROWS; ++row) {
        for (uint32_t col = 0U; col < COLS; ++col) {
            const uint32_t index = row * COLS + col;
            const float src_value =
                (static_cast<int32_t>(row % 19U) - 9) * 0.125f +
                (static_cast<int32_t>(col % 17U) - 8) * 0.03125f;
            const float rhs_value =
                (static_cast<int32_t>(row % 13U) - 6) * 0.0625f -
                (static_cast<int32_t>(col % 11U) - 5) * 0.046875f;
            src[index] = npu_float_to_bits(src_value);
            rhs[index] = npu_float_to_bits(rhs_value);
        }
    }

    llm_clear_slot_span(SRC_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(RHS_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(EXP_DST_SLOT, MAT_SLOT_SPAN);
    llm_clear_slot_span(ADD_DST_SLOT, MAT_SLOT_SPAN);

    llm_store_logical_matrix_last_axis_front(SRC_SLOT, src.data(), ROWS, COLS);
    llm_store_logical_matrix_last_axis_front(RHS_SLOT, rhs.data(), ROWS, COLS);
    npu_golden_vpu_unary_exp(src.data(), exp_expected.data(), ROWS * COLS);
    npu_golden_vpu_elemwise_f32(VPU_OP_VADD, src.data(), rhs.data(),
                                add_expected.data(),
                                ROWS * COLS);

    const PrimitiveTensorDesc src_desc =
        llm_packed_last_axis_tensor(SRC_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc rhs_desc =
        llm_packed_last_axis_tensor(RHS_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc exp_dst_desc =
        llm_packed_last_axis_tensor(EXP_DST_SLOT, ROWS, COLS);
    const PrimitiveTensorDesc add_dst_desc =
        llm_packed_last_axis_tensor(ADD_DST_SLOT, ROWS, COLS);
    const size_t unary_count = vpu_primitive_unary(
        VPU_DEVICE_ID, VPU_OP_VEXP, src_desc, exp_dst_desc, {0U, 1U, 0U},
        EXP_SYNC);
    npu_launch_sync_wait(VPU_DEVICE_ID, EXP_SYNC, 0U, 0U, 0U);
    const size_t binary_count = vpu_primitive_binary(
        VPU_DEVICE_ID, VPU_OP_VADD, src_desc, rhs_desc, add_dst_desc,
        {2U, 3U, 1U}, ADD_SYNC);
    npu_launch_sync_wait(VPU_DEVICE_ID, ADD_SYNC, 0U, 0U, 0U);
    npu_cmd_sync_done();

    llm_load_logical_matrix_last_axis_front(EXP_DST_SLOT, actual.data(), ROWS,
                                            COLS);
    if (npu_expect_float_vector_close(
            "BUILDER_SMOKE_UNARY_EXP", exp_expected.data(), actual.data(),
            ROWS * COLS,
            0.03f, 0.03f) != 0) {
        printf("BUILDER_SMOKE_F32_FAIL\n");
        return 1;
    }
    llm_load_logical_matrix_last_axis_front(ADD_DST_SLOT, actual.data(), ROWS,
                                            COLS);
    if (npu_expect_float_vector_close(
            "BUILDER_SMOKE_BINARY_ADD", add_expected.data(), actual.data(),
            ROWS * COLS,
            0.01f, 0.01f) != 0) {
        printf("BUILDER_SMOKE_F32_FAIL\n");
        return 1;
    }

    printf("BUILDER_SMOKE_UNARY_CACHED_CMDS=%u\n", (unsigned)unary_count);
    printf("BUILDER_SMOKE_BINARY_CACHED_CMDS=%u\n", (unsigned)binary_count);
    printf("BUILDER_SMOKE_F32_PASS\n");
    return 0;
}
