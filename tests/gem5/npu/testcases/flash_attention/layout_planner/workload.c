/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include <array>

#include "flash_attention_layout.hh"

namespace
{

static uintptr_t
manualAlignUp(uintptr_t value, uint32_t alignment_bytes)
{
    const uintptr_t mask = (uintptr_t)alignment_bytes - 1U;
    return (value + mask) & ~mask;
}

static int
expectEqualU32(const char *label, uint32_t actual, uint32_t expected)
{
    if (actual == expected) {
        return 0;
    }

    printf("%s actual=%u expected=%u\n", label, actual, expected);
    return 1;
}

static int
expectEqualAddr(const char *label, uintptr_t actual, uintptr_t expected)
{
    if (actual == expected) {
        return 0;
    }

    printf("%s actual=%#llx expected=%#llx\n", label,
           (unsigned long long)actual, (unsigned long long)expected);
    return 1;
}

static int
expectStatus(const char *label, int actual, int expected)
{
    if (actual == expected) {
        return 0;
    }

    printf("%s actual=%d expected=%d\n", label, actual, expected);
    return 1;
}

static int
expectNoOverlap(const char *lhs_name, const NpuFlashAttentionRegion *lhs,
                const char *rhs_name, const NpuFlashAttentionRegion *rhs)
{
    if (!npu_flash_attention_regions_overlap(lhs, rhs)) {
        return 0;
    }

    printf("OVERLAP lhs=%s[%#llx,%#llx) rhs=%s[%#llx,%#llx)\n", lhs_name,
           (unsigned long long)lhs->base_addr,
           (unsigned long long)(lhs->base_addr + lhs->bytes), rhs_name,
           (unsigned long long)rhs->base_addr,
           (unsigned long long)(rhs->base_addr + rhs->bytes));
    return 1;
}

static int
checkExpectedAddresses(const NpuFlashAttentionPlannerRequest &request,
                       const NpuFlashAttentionPlannerOutput &layout)
{
    uintptr_t cursor = request.workspace_base_addr;
    uintptr_t expected = 0U;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("Q_BASE", layout.q_i8.region.base_addr, expected)) {
        return 1;
    }
    cursor = expected + layout.q_i8.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("K_BASE", layout.k_i8.region.base_addr, expected)) {
        return 1;
    }
    cursor = expected + layout.k_i8.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("K_T_BASE", layout.k_t_i8.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.k_t_i8.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("V_BASE", layout.v_i8.region.base_addr, expected)) {
        return 1;
    }
    cursor = expected + layout.v_i8.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("SCORES_I32_BASE", layout.scores_i32.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.scores_i32.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("SCORES_F32_BASE", layout.scores_f32.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.scores_f32.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("M_STATE_BASE", layout.m_state.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.m_state.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("L_STATE_BASE", layout.l_state.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.l_state.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("P_F32_BASE", layout.p_f32.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.p_f32.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("P_I8_BASE", layout.p_i8.region.base_addr, expected)) {
        return 1;
    }
    cursor = expected + layout.p_i8.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("OUT_I32_BASE", layout.out_i32.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.out_i32.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("OUT_F32_BASE", layout.out_f32.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.out_f32.region.bytes;

    expected = manualAlignUp(cursor, request.alignment_bytes);
    if (expectEqualAddr("SCALE_BASE", layout.scale.region.base_addr,
                        expected)) {
        return 1;
    }
    cursor = expected + layout.scale.region.bytes;

    return expectEqualU32("USED_BYTES", layout.used_bytes,
                          (uint32_t)(cursor - request.workspace_base_addr));
}

static int
checkDenseShapes(const NpuFlashAttentionPlannerRequest &request,
                 const NpuFlashAttentionPlannerOutput &layout)
{
    if (expectEqualU32("Q_ROWS", layout.q_i8.rows, request.seq_q) ||
        expectEqualU32("Q_COLS", layout.q_i8.cols, request.dim) ||
        expectEqualU32("Q_STRIDE", layout.q_i8.row_stride_bytes,
                       request.dim) ||
        expectEqualU32("K_ROWS", layout.k_i8.rows, request.seq_k) ||
        expectEqualU32("K_COLS", layout.k_i8.cols, request.dim) ||
        expectEqualU32("K_T_ROWS", layout.k_t_i8.rows, request.dim) ||
        expectEqualU32("K_T_COLS", layout.k_t_i8.cols, request.seq_k) ||
        expectEqualU32("V_ROWS", layout.v_i8.rows, request.seq_k) ||
        expectEqualU32("V_COLS", layout.v_i8.cols, request.dim_v) ||
        expectEqualU32("SCORES_I32_STRIDE", layout.scores_i32.row_stride_bytes,
                       request.seq_k * sizeof(int32_t)) ||
        expectEqualU32("SCORES_F32_STRIDE", layout.scores_f32.row_stride_bytes,
                       request.seq_k * sizeof(float)) ||
        expectEqualU32("M_STATE_LEN", layout.m_state.length, request.seq_q) ||
        expectEqualU32("L_STATE_LEN", layout.l_state.length, request.seq_q) ||
        expectEqualU32("P_F32_STRIDE", layout.p_f32.row_stride_bytes,
                       request.seq_k * sizeof(float)) ||
        expectEqualU32("P_I8_STRIDE", layout.p_i8.row_stride_bytes,
                       request.seq_k) ||
        expectEqualU32("OUT_I32_STRIDE", layout.out_i32.row_stride_bytes,
                       request.dim_v * sizeof(int32_t)) ||
        expectEqualU32("OUT_F32_STRIDE", layout.out_f32.row_stride_bytes,
                       request.dim_v * sizeof(float)) ||
        expectEqualU32("SCALE_BYTES", layout.scale.region.bytes,
                       sizeof(float))) {
        return 1;
    }

    return 0;
}

static int
checkAddressing(const NpuFlashAttentionPlannerOutput &layout)
{
    const uintptr_t q_addr =
        npu_flash_attention_matrix_addr(&layout.q_i8, 2U, 3U);
    const uintptr_t q_expected =
        layout.q_i8.region.base_addr +
        (uintptr_t)2U * layout.q_i8.row_stride_bytes + 3U;
    if (expectEqualAddr("Q_ROW_MAJOR_ADDR", q_addr, q_expected)) {
        return 1;
    }

    const uintptr_t k_addr =
        npu_flash_attention_matrix_addr(&layout.k_i8, 4U, 2U);
    const uintptr_t k_expected =
        layout.k_i8.region.base_addr +
        (uintptr_t)4U * layout.k_i8.row_stride_bytes + 2U;
    if (expectEqualAddr("K_ROW_MAJOR_ADDR", k_addr, k_expected)) {
        return 1;
    }

    const uintptr_t scores_addr =
        npu_flash_attention_matrix_addr(&layout.scores_i32, 1U, 4U);
    const uintptr_t scores_expected =
        layout.scores_i32.region.base_addr +
        (uintptr_t)1U * layout.scores_i32.row_stride_bytes +
        (uintptr_t)4U * sizeof(int32_t);
    if (expectEqualAddr("SCORES_I32_ROW_MAJOR_ADDR", scores_addr,
                        scores_expected)) {
        return 1;
    }

    const uintptr_t out_addr =
        npu_flash_attention_matrix_addr(&layout.out_f32, 2U, 5U);
    const uintptr_t out_expected =
        layout.out_f32.region.base_addr +
        (uintptr_t)2U * layout.out_f32.row_stride_bytes +
        (uintptr_t)5U * sizeof(float);
    if (expectEqualAddr("OUT_F32_ROW_MAJOR_ADDR", out_addr, out_expected)) {
        return 1;
    }

    const uintptr_t m_state_addr =
        npu_flash_attention_vector_addr(&layout.m_state, 1U);
    const uintptr_t m_state_expected =
        layout.m_state.region.base_addr + sizeof(float);
    if (expectEqualAddr("M_STATE_ADDR", m_state_addr, m_state_expected)) {
        return 1;
    }

    const uintptr_t k_t_addr =
        npu_flash_attention_matrix_transpose_of_addr(&layout.k_t_i8, 4U, 2U);
    const uintptr_t k_t_expected =
        layout.k_t_i8.region.base_addr +
        (uintptr_t)2U * layout.k_t_i8.row_stride_bytes + 4U;
    if (expectEqualAddr("K_T_TRANSPOSE_ADDR", k_t_addr, k_t_expected)) {
        return 1;
    }

    return 0;
}

static int
checkNoOverlap(const NpuFlashAttentionPlannerOutput &layout)
{
    struct NamedRegion
    {
        const char *name;
        const NpuFlashAttentionRegion *region;
    };

    const std::array<NamedRegion, 13> regions = {{
        {"Q_i8", &layout.q_i8.region},
        {"K_i8", &layout.k_i8.region},
        {"K_t_i8", &layout.k_t_i8.region},
        {"V_i8", &layout.v_i8.region},
        {"scores_i32", &layout.scores_i32.region},
        {"scores_f32", &layout.scores_f32.region},
        {"m_state", &layout.m_state.region},
        {"l_state", &layout.l_state.region},
        {"p_f32", &layout.p_f32.region},
        {"p_i8", &layout.p_i8.region},
        {"out_i32", &layout.out_i32.region},
        {"out_f32", &layout.out_f32.region},
        {"scale", &layout.scale.region},
    }};

    for (size_t lhs = 0U; lhs < regions.size(); ++lhs) {
        for (size_t rhs = lhs + 1U; rhs < regions.size(); ++rhs) {
            if (expectNoOverlap(regions[lhs].name, regions[lhs].region,
                                regions[rhs].name, regions[rhs].region)) {
                return 1;
            }
        }
    }

    return 0;
}

} // namespace

int
main(void)
{
    NpuFlashAttentionPlannerRequest request = {};
    request.workspace_base_addr = 0x60020003ULL;
    request.workspace_bytes = 4096U;
    request.seq_q = 3U;
    request.seq_k = 5U;
    request.dim = 4U;
    request.dim_v = 6U;
    request.alignment_bytes = 64U;

    NpuFlashAttentionPlannerOutput layout = {};
    int status = npu_flash_attention_plan_layout(&request, &layout);
    if (expectStatus("PLAN_STATUS", status, NPU_FLASH_ATTENTION_PLANNER_OK)) {
        return 1;
    }
    const uint32_t used_bytes = layout.used_bytes;
    if (expectStatus("VALIDATE_STATUS",
                     npu_flash_attention_validate_output(&request, &layout),
                     NPU_FLASH_ATTENTION_PLANNER_OK)) {
        return 1;
    }

    if (checkExpectedAddresses(request, layout) ||
        checkDenseShapes(request, layout) || checkAddressing(layout) ||
        checkNoOverlap(layout)) {
        return 1;
    }

    NpuFlashAttentionPlannerRequest too_small_request = request;
    too_small_request.workspace_bytes = used_bytes - 1U;
    status = npu_flash_attention_plan_layout(&too_small_request, &layout);
    if (expectStatus("PLAN_TOO_SMALL", status,
                     NPU_FLASH_ATTENTION_PLANNER_INSUFFICIENT_WORKSPACE)) {
        return 1;
    }

    printf("FLASH_ATTENTION_LAYOUT_PLANNER_PASS used_bytes=%u\n",
           used_bytes);
    return 0;
}
