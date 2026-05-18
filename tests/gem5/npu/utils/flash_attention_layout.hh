#ifndef TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_LAYOUT_HH_
#define TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_LAYOUT_HH_

#include <stdint.h>

#include <cstddef>
#include <cstdio>

/*
 * FlashAttention Stage 3 layout/scratch contract:
 *
 * - Every buffer lives inside one caller-provided workspace window.
 * - Every matrix/vector/scalar is dense.
 * - Matrix buffers use row-major byte addressing.
 * - K_t_i8 stores K^T densely, so logical K(row, col) maps to
 *   K_t_i8(col, row).
 * - The planner validates shape/stride contracts and rejects overlapping
 *   regions.
 */

enum NpuFlashAttentionPlannerStatus
{
    NPU_FLASH_ATTENTION_PLANNER_OK = 0,
    NPU_FLASH_ATTENTION_PLANNER_INVALID_ARGUMENT = -1,
    NPU_FLASH_ATTENTION_PLANNER_SIZE_OVERFLOW = -2,
    NPU_FLASH_ATTENTION_PLANNER_INSUFFICIENT_WORKSPACE = -3,
    NPU_FLASH_ATTENTION_PLANNER_LAYOUT_INVALID = -4,
};

struct NpuFlashAttentionPlannerRequest
{
    uintptr_t workspace_base_addr = 0U;
    uint32_t workspace_bytes = 0U;
    uint32_t seq_q = 0U;
    uint32_t seq_k = 0U;
    uint32_t dim = 0U;
    uint32_t dim_v = 0U;
    uint32_t alignment_bytes = 64U;
};

struct NpuFlashAttentionRegion
{
    uintptr_t base_addr = 0U;
    uint32_t bytes = 0U;
};

struct NpuFlashAttentionMatrixLayout
{
    NpuFlashAttentionRegion region = {};
    uint32_t rows = 0U;
    uint32_t cols = 0U;
    uint32_t elem_bytes = 0U;
    uint32_t row_stride_bytes = 0U;
};

struct NpuFlashAttentionVectorLayout
{
    NpuFlashAttentionRegion region = {};
    uint32_t length = 0U;
    uint32_t elem_bytes = 0U;
    uint32_t stride_bytes = 0U;
};

struct NpuFlashAttentionScalarLayout
{
    NpuFlashAttentionRegion region = {};
    uint32_t elem_bytes = 0U;
};

struct NpuFlashAttentionPlannerOutput
{
    uintptr_t workspace_base_addr = 0U;
    uint32_t workspace_bytes = 0U;
    uint32_t used_bytes = 0U;

    NpuFlashAttentionMatrixLayout q_i8 = {};
    NpuFlashAttentionMatrixLayout k_i8 = {};
    NpuFlashAttentionMatrixLayout k_t_i8 = {};
    NpuFlashAttentionMatrixLayout v_i8 = {};
    NpuFlashAttentionMatrixLayout scores_i32 = {};
    NpuFlashAttentionMatrixLayout scores_f32 = {};
    NpuFlashAttentionVectorLayout m_state = {};
    NpuFlashAttentionVectorLayout l_state = {};
    NpuFlashAttentionMatrixLayout p_f32 = {};
    NpuFlashAttentionMatrixLayout p_i8 = {};
    NpuFlashAttentionMatrixLayout out_i32 = {};
    NpuFlashAttentionMatrixLayout out_f32 = {};
    NpuFlashAttentionScalarLayout scale = {};
};

static inline int
npu_flash_attention_alignment_valid(uint32_t alignment_bytes)
{
    return alignment_bytes != 0U &&
           (alignment_bytes & (alignment_bytes - 1U)) == 0U;
}

static inline uintptr_t
npu_flash_attention_align_up(uintptr_t value, uint32_t alignment_bytes)
{
    const uintptr_t mask = (uintptr_t)alignment_bytes - 1U;
    return (value + mask) & ~mask;
}

static inline uint64_t
npu_flash_attention_matrix_dense_bytes_u64(uint32_t rows, uint32_t cols,
                                           uint32_t elem_bytes)
{
    return (uint64_t)rows * (uint64_t)cols * (uint64_t)elem_bytes;
}

static inline uint64_t
npu_flash_attention_vector_dense_bytes_u64(uint32_t length,
                                           uint32_t elem_bytes)
{
    return (uint64_t)length * (uint64_t)elem_bytes;
}

static inline uintptr_t
npu_flash_attention_matrix_addr(const NpuFlashAttentionMatrixLayout *layout,
                                uint32_t row, uint32_t col)
{
    return layout->region.base_addr +
           (uintptr_t)row * (uintptr_t)layout->row_stride_bytes +
           (uintptr_t)col * (uintptr_t)layout->elem_bytes;
}

static inline uintptr_t
npu_flash_attention_matrix_transpose_of_addr(
    const NpuFlashAttentionMatrixLayout *transpose_layout,
    uint32_t src_row, uint32_t src_col)
{
    return npu_flash_attention_matrix_addr(transpose_layout, src_col, src_row);
}

static inline uintptr_t
npu_flash_attention_vector_addr(const NpuFlashAttentionVectorLayout *layout,
                                uint32_t index)
{
    return layout->region.base_addr +
           (uintptr_t)index * (uintptr_t)layout->stride_bytes;
}

static inline int
npu_flash_attention_regions_overlap(const NpuFlashAttentionRegion *lhs,
                                    const NpuFlashAttentionRegion *rhs)
{
    if (lhs->bytes == 0U || rhs->bytes == 0U) {
        return 0;
    }

    const uintptr_t lhs_end = lhs->base_addr + (uintptr_t)lhs->bytes;
    const uintptr_t rhs_end = rhs->base_addr + (uintptr_t)rhs->bytes;
    return lhs->base_addr < rhs_end && rhs->base_addr < lhs_end;
}

static inline int
npu_flash_attention_validate_matrix_layout(
    const NpuFlashAttentionMatrixLayout *layout)
{
    const uint64_t expected_bytes =
        npu_flash_attention_matrix_dense_bytes_u64(layout->rows, layout->cols,
                                                   layout->elem_bytes);

    if (layout->rows == 0U || layout->cols == 0U || layout->elem_bytes == 0U) {
        return 0;
    }
    if (layout->row_stride_bytes != layout->cols * layout->elem_bytes) {
        return 0;
    }
    if (layout->region.bytes != expected_bytes) {
        return 0;
    }
    return 1;
}

static inline int
npu_flash_attention_validate_vector_layout(
    const NpuFlashAttentionVectorLayout *layout)
{
    const uint64_t expected_bytes =
        npu_flash_attention_vector_dense_bytes_u64(layout->length,
                                                   layout->elem_bytes);

    if (layout->length == 0U || layout->elem_bytes == 0U) {
        return 0;
    }
    if (layout->stride_bytes != layout->elem_bytes) {
        return 0;
    }
    if (layout->region.bytes != expected_bytes) {
        return 0;
    }
    return 1;
}

static inline int
npu_flash_attention_validate_scalar_layout(
    const NpuFlashAttentionScalarLayout *layout)
{
    return layout->elem_bytes != 0U &&
           layout->region.bytes == layout->elem_bytes;
}

static inline uintptr_t
npu_flash_attention_reserve_region(uintptr_t *cursor, uint32_t bytes,
                                   uint32_t alignment_bytes)
{
    const uintptr_t base = npu_flash_attention_align_up(*cursor,
                                                        alignment_bytes);
    *cursor = base + (uintptr_t)bytes;
    return base;
}

static inline void
npu_flash_attention_assign_matrix(NpuFlashAttentionMatrixLayout *layout,
                                  uintptr_t *cursor, uint32_t rows,
                                  uint32_t cols, uint32_t elem_bytes,
                                  uint32_t alignment_bytes)
{
    const uint32_t bytes =
        (uint32_t)npu_flash_attention_matrix_dense_bytes_u64(
            rows, cols, elem_bytes);
    layout->rows = rows;
    layout->cols = cols;
    layout->elem_bytes = elem_bytes;
    layout->row_stride_bytes = cols * elem_bytes;
    layout->region.base_addr =
        npu_flash_attention_reserve_region(cursor, bytes, alignment_bytes);
    layout->region.bytes = bytes;
}

static inline void
npu_flash_attention_assign_vector(NpuFlashAttentionVectorLayout *layout,
                                  uintptr_t *cursor, uint32_t length,
                                  uint32_t elem_bytes,
                                  uint32_t alignment_bytes)
{
    const uint32_t bytes =
        (uint32_t)npu_flash_attention_vector_dense_bytes_u64(length,
                                                             elem_bytes);
    layout->length = length;
    layout->elem_bytes = elem_bytes;
    layout->stride_bytes = elem_bytes;
    layout->region.base_addr =
        npu_flash_attention_reserve_region(cursor, bytes, alignment_bytes);
    layout->region.bytes = bytes;
}

static inline void
npu_flash_attention_assign_scalar(NpuFlashAttentionScalarLayout *layout,
                                  uintptr_t *cursor, uint32_t elem_bytes,
                                  uint32_t alignment_bytes)
{
    layout->elem_bytes = elem_bytes;
    layout->region.base_addr =
        npu_flash_attention_reserve_region(cursor, elem_bytes,
                                           alignment_bytes);
    layout->region.bytes = elem_bytes;
}

static inline int
npu_flash_attention_validate_output(
    const NpuFlashAttentionPlannerRequest *request,
    const NpuFlashAttentionPlannerOutput *output)
{
    const NpuFlashAttentionRegion *regions[] = {
        &output->q_i8.region,       &output->k_i8.region,
        &output->k_t_i8.region,     &output->v_i8.region,
        &output->scores_i32.region, &output->scores_f32.region,
        &output->m_state.region,    &output->l_state.region,
        &output->p_f32.region,      &output->p_i8.region,
        &output->out_i32.region,    &output->out_f32.region,
        &output->scale.region,
    };

    if (!npu_flash_attention_validate_matrix_layout(&output->q_i8) ||
        !npu_flash_attention_validate_matrix_layout(&output->k_i8) ||
        !npu_flash_attention_validate_matrix_layout(&output->k_t_i8) ||
        !npu_flash_attention_validate_matrix_layout(&output->v_i8) ||
        !npu_flash_attention_validate_matrix_layout(&output->scores_i32) ||
        !npu_flash_attention_validate_matrix_layout(&output->scores_f32) ||
        !npu_flash_attention_validate_vector_layout(&output->m_state) ||
        !npu_flash_attention_validate_vector_layout(&output->l_state) ||
        !npu_flash_attention_validate_matrix_layout(&output->p_f32) ||
        !npu_flash_attention_validate_matrix_layout(&output->p_i8) ||
        !npu_flash_attention_validate_matrix_layout(&output->out_i32) ||
        !npu_flash_attention_validate_matrix_layout(&output->out_f32) ||
        !npu_flash_attention_validate_scalar_layout(&output->scale)) {
        return NPU_FLASH_ATTENTION_PLANNER_LAYOUT_INVALID;
    }

    if (output->q_i8.rows != request->seq_q ||
        output->q_i8.cols != request->dim ||
        output->k_i8.rows != request->seq_k ||
        output->k_i8.cols != request->dim ||
        output->k_t_i8.rows != request->dim ||
        output->k_t_i8.cols != request->seq_k ||
        output->v_i8.rows != request->seq_k ||
        output->v_i8.cols != request->dim_v ||
        output->scores_i32.rows != request->seq_q ||
        output->scores_i32.cols != request->seq_k ||
        output->scores_f32.rows != request->seq_q ||
        output->scores_f32.cols != request->seq_k ||
        output->m_state.length != request->seq_q ||
        output->l_state.length != request->seq_q ||
        output->p_f32.rows != request->seq_q ||
        output->p_f32.cols != request->seq_k ||
        output->p_i8.rows != request->seq_q ||
        output->p_i8.cols != request->seq_k ||
        output->out_i32.rows != request->seq_q ||
        output->out_i32.cols != request->dim_v ||
        output->out_f32.rows != request->seq_q ||
        output->out_f32.cols != request->dim_v ||
        output->scale.elem_bytes != sizeof(float)) {
        return NPU_FLASH_ATTENTION_PLANNER_LAYOUT_INVALID;
    }

    for (size_t lhs = 0U; lhs < sizeof(regions) / sizeof(regions[0]); ++lhs) {
        for (size_t rhs = lhs + 1U;
             rhs < sizeof(regions) / sizeof(regions[0]); ++rhs) {
            if (npu_flash_attention_regions_overlap(regions[lhs],
                                                    regions[rhs])) {
                return NPU_FLASH_ATTENTION_PLANNER_LAYOUT_INVALID;
            }
        }
    }

    if (output->used_bytes > request->workspace_bytes) {
        return NPU_FLASH_ATTENTION_PLANNER_INSUFFICIENT_WORKSPACE;
    }

    return NPU_FLASH_ATTENTION_PLANNER_OK;
}

static inline int
npu_flash_attention_plan_layout(
    const NpuFlashAttentionPlannerRequest *request,
    NpuFlashAttentionPlannerOutput *output)
{
    if (request == NULL || output == NULL) {
        return NPU_FLASH_ATTENTION_PLANNER_INVALID_ARGUMENT;
    }
    if (request->seq_q == 0U || request->seq_k == 0U || request->dim == 0U ||
        request->dim_v == 0U ||
        !npu_flash_attention_alignment_valid(request->alignment_bytes)) {
        return NPU_FLASH_ATTENTION_PLANNER_INVALID_ARGUMENT;
    }

    const uint64_t checked_sizes[] = {
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_q,
                                                   request->dim,
                                                   sizeof(int8_t)),
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_k,
                                                   request->dim,
                                                   sizeof(int8_t)),
        npu_flash_attention_matrix_dense_bytes_u64(request->dim,
                                                   request->seq_k,
                                                   sizeof(int8_t)),
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_k,
                                                   request->dim_v,
                                                   sizeof(int8_t)),
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_q,
                                                   request->seq_k,
                                                   sizeof(int32_t)),
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_q,
                                                   request->seq_k,
                                                   sizeof(float)),
        npu_flash_attention_vector_dense_bytes_u64(request->seq_q,
                                                   sizeof(float)),
        npu_flash_attention_vector_dense_bytes_u64(request->seq_q,
                                                   sizeof(float)),
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_q,
                                                   request->seq_k,
                                                   sizeof(float)),
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_q,
                                                   request->seq_k,
                                                   sizeof(int8_t)),
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_q,
                                                   request->dim_v,
                                                   sizeof(int32_t)),
        npu_flash_attention_matrix_dense_bytes_u64(request->seq_q,
                                                   request->dim_v,
                                                   sizeof(float)),
        sizeof(float),
    };

    for (size_t idx = 0U;
         idx < sizeof(checked_sizes) / sizeof(checked_sizes[0]);
         ++idx) {
        if (checked_sizes[idx] > UINT32_MAX) {
            return NPU_FLASH_ATTENTION_PLANNER_SIZE_OVERFLOW;
        }
    }

    *output = {};
    output->workspace_base_addr = request->workspace_base_addr;
    output->workspace_bytes = request->workspace_bytes;

    uintptr_t cursor = request->workspace_base_addr;
    npu_flash_attention_assign_matrix(&output->q_i8, &cursor, request->seq_q,
                                      request->dim, sizeof(int8_t),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->k_i8, &cursor, request->seq_k,
                                      request->dim, sizeof(int8_t),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->k_t_i8, &cursor, request->dim,
                                      request->seq_k, sizeof(int8_t),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->v_i8, &cursor, request->seq_k,
                                      request->dim_v, sizeof(int8_t),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->scores_i32, &cursor,
                                      request->seq_q, request->seq_k,
                                      sizeof(int32_t),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->scores_f32, &cursor,
                                      request->seq_q, request->seq_k,
                                      sizeof(float),
                                      request->alignment_bytes);
    npu_flash_attention_assign_vector(&output->m_state, &cursor,
                                      request->seq_q, sizeof(float),
                                      request->alignment_bytes);
    npu_flash_attention_assign_vector(&output->l_state, &cursor,
                                      request->seq_q, sizeof(float),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->p_f32, &cursor, request->seq_q,
                                      request->seq_k, sizeof(float),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->p_i8, &cursor, request->seq_q,
                                      request->seq_k, sizeof(int8_t),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->out_i32, &cursor,
                                      request->seq_q, request->dim_v,
                                      sizeof(int32_t),
                                      request->alignment_bytes);
    npu_flash_attention_assign_matrix(&output->out_f32, &cursor,
                                      request->seq_q, request->dim_v,
                                      sizeof(float),
                                      request->alignment_bytes);
    npu_flash_attention_assign_scalar(&output->scale, &cursor, sizeof(float),
                                      request->alignment_bytes);

    const uintptr_t workspace_end = cursor;
    const uintptr_t used = workspace_end - request->workspace_base_addr;
    if (used > request->workspace_bytes || used > UINT32_MAX) {
        return NPU_FLASH_ATTENTION_PLANNER_INSUFFICIENT_WORKSPACE;
    }

    output->used_bytes = (uint32_t)used;
    return npu_flash_attention_validate_output(request, output);
}

#endif
