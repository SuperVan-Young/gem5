/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "llm_test_utils.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "binary.hh"
#include "unary.hh"
#include "vpu_elemwise.hh"
#include "vpu_unary.hh"

enum BuilderSmokeLayout
{
    VPU_DEVICE_ID = 0U,
    ROWS = 2U,
    COLS = 8U,
    SRC_SLOT = 0U,
    RHS_SLOT = 1U,
    EXP_DST_SLOT = 2U,
    ADD_DST_SLOT = 3U,
    EXP_SYNC = 0x41U,
    ADD_SYNC = 0x42U,
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
    const uint32_t rhs[ROWS * COLS] = {
        npu_float_to_bits(0.5f),  npu_float_to_bits(0.25f),
        npu_float_to_bits(-0.5f), npu_float_to_bits(-0.25f),
        npu_float_to_bits(1.0f),  npu_float_to_bits(1.5f),
        npu_float_to_bits(-1.0f), npu_float_to_bits(-1.5f),
        npu_float_to_bits(0.75f), npu_float_to_bits(-0.75f),
        npu_float_to_bits(1.25f), npu_float_to_bits(-1.25f),
        npu_float_to_bits(1.75f), npu_float_to_bits(-1.75f),
        npu_float_to_bits(0.125f), npu_float_to_bits(-0.125f),
    };
    uint32_t exp_expected[ROWS * COLS];
    uint32_t add_expected[ROWS * COLS];
    uint32_t actual[ROWS * COLS];

    for (uint32_t slot = 0U; slot < 4U; ++slot) {
        npu_spm_clear_slot(slot);
    }

    llm_store_logical_matrix_last_axis_front(SRC_SLOT, src, ROWS, COLS);
    llm_store_logical_matrix_last_axis_front(RHS_SLOT, rhs, ROWS, COLS);
    npu_golden_vpu_unary_exp(src, exp_expected, ROWS * COLS);
    npu_golden_vpu_elemwise_f32(VPU_OP_VADD, src, rhs, add_expected,
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

    llm_load_logical_matrix_last_axis_front(EXP_DST_SLOT, actual, ROWS, COLS);
    if (npu_expect_float_vector_close(
            "BUILDER_SMOKE_UNARY_EXP", exp_expected, actual, ROWS * COLS,
            0.03f, 0.03f) != 0) {
        printf("BUILDER_SMOKE_F32_FAIL\n");
        return 1;
    }
    llm_load_logical_matrix_last_axis_front(ADD_DST_SLOT, actual, ROWS, COLS);
    if (npu_expect_float_vector_close(
            "BUILDER_SMOKE_BINARY_ADD", add_expected, actual, ROWS * COLS,
            0.01f, 0.01f) != 0) {
        printf("BUILDER_SMOKE_F32_FAIL\n");
        return 1;
    }

    printf("BUILDER_SMOKE_UNARY_CACHED_CMDS=%u\n", (unsigned)unary_count);
    printf("BUILDER_SMOKE_BINARY_CACHED_CMDS=%u\n", (unsigned)binary_count);
    printf("BUILDER_SMOKE_F32_PASS\n");
    return 0;
}
