#ifndef TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_HH_
#define TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_HH_

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "flash_attention_pv.hh"
#include "flash_attention_qk.hh"
#include "mpu_gemm.hh"
#include "npu_assert.hh"
#include "primitive/online_softmax.hh"

typedef struct
{
    uint32_t seq_q;
    uint32_t seq_k;
    uint32_t dim;
    uint32_t dim_v;
} NpuFlashAttentionShape;

typedef struct
{
    const float *q_f32;
    const float *k_f32;
    const float *v_f32;
    const int32_t *expected_scores_i32;
    const int32_t *expected_out_i32;
    int8_t *q_i8;
    int8_t *k_i8;
    int8_t *k_t_i8;
    int8_t *v_i8;
    int32_t *scores_i32;
    float *scores_f32;
    float *m_state;
    float *l_state;
    float *p_f32;
    int8_t *p_i8;
    int32_t *out_i32;
    float *out_f32;
    PrimitiveScaleMetadata *q_metadata;
    PrimitiveScaleMetadata *k_metadata;
    PrimitiveScaleMetadata *p_metadata;
    PrimitiveScaleMetadata *v_metadata;
    int8_t *k_t_tile_i8;
    int8_t *p_tile_i8;
    int8_t *v_tile_i8;
    int32_t *tile_i32;
} NpuFlashAttentionBuffers;

typedef struct
{
    NpuMpuGemmSpmI8Matrix q_matrix;
    NpuMpuGemmSpmI8Matrix k_t_matrix;
    NpuMpuGemmSpmI32Matrix scores_matrix;
    NpuMpuGemmSpmI8Matrix p_matrix;
    NpuMpuGemmSpmI8Matrix v_matrix;
    NpuMpuGemmSpmI32Matrix out_matrix;
    NpuMpuGemmTileShape qk_tile;
    NpuMpuGemmTileShape pv_tile;
    NpuMpuGemmLaunchConfig qk_launch;
    NpuMpuGemmLaunchConfig pv_launch;
    uint64_t wait_timeout;
} NpuFlashAttentionExecutionConfig;

enum NpuFlashAttentionStatus
{
    NPU_FLASH_ATTENTION_OK = 0,
    NPU_FLASH_ATTENTION_INVALID_ARGUMENT = -1,
    NPU_FLASH_ATTENTION_QK_LAUNCH_FAILED = -2,
    NPU_FLASH_ATTENTION_QK_WAIT_FAILED = -3,
    NPU_FLASH_ATTENTION_PV_LAUNCH_FAILED = -4,
    NPU_FLASH_ATTENTION_PV_WAIT_FAILED = -5,
};

static inline NpuFlashAttentionQkShape
npu_flash_attention_qk_shape_from_basic(
    const NpuFlashAttentionShape *shape)
{
    const NpuFlashAttentionQkShape qk_shape = {
        shape->seq_q,
        shape->seq_k,
        shape->dim,
    };
    return qk_shape;
}

static inline NpuFlashAttentionPvShape
npu_flash_attention_pv_shape_from_basic(
    const NpuFlashAttentionShape *shape)
{
    const NpuFlashAttentionPvShape pv_shape = {
        shape->seq_q,
        shape->seq_k,
        shape->dim_v,
    };
    return pv_shape;
}

static inline int
npu_flash_attention_validate_basic_arguments(
    const NpuFlashAttentionShape *shape,
    const NpuFlashAttentionBuffers *buffers,
    const NpuFlashAttentionExecutionConfig *config)
{
    if (shape == NULL || buffers == NULL || config == NULL) {
        return NPU_FLASH_ATTENTION_INVALID_ARGUMENT;
    }
    if (shape->seq_q == 0U || shape->seq_k == 0U || shape->dim == 0U ||
        shape->dim_v == 0U) {
        return NPU_FLASH_ATTENTION_INVALID_ARGUMENT;
    }
    if (buffers->q_f32 == NULL || buffers->k_f32 == NULL ||
        buffers->v_f32 == NULL || buffers->q_i8 == NULL ||
        buffers->k_i8 == NULL || buffers->k_t_i8 == NULL ||
        buffers->v_i8 == NULL || buffers->scores_i32 == NULL ||
        buffers->scores_f32 == NULL || buffers->m_state == NULL ||
        buffers->l_state == NULL || buffers->p_f32 == NULL ||
        buffers->p_i8 == NULL || buffers->out_i32 == NULL ||
        buffers->out_f32 == NULL || buffers->q_metadata == NULL ||
        buffers->k_metadata == NULL || buffers->p_metadata == NULL ||
        buffers->v_metadata == NULL || buffers->k_t_tile_i8 == NULL ||
        buffers->p_tile_i8 == NULL || buffers->v_tile_i8 == NULL ||
        buffers->tile_i32 == NULL) {
        return NPU_FLASH_ATTENTION_INVALID_ARGUMENT;
    }
    if (config->wait_timeout == 0ULL) {
        return NPU_FLASH_ATTENTION_INVALID_ARGUMENT;
    }

    return NPU_FLASH_ATTENTION_OK;
}

static inline void
npu_flash_attention_copy_i8_block(const int8_t *src, uint32_t src_cols,
                                  uint32_t row0, uint32_t col0,
                                  uint32_t rows, uint32_t cols, int8_t *dst)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        memcpy(dst + row * cols, src + (row0 + row) * src_cols + col0,
               cols * sizeof(int8_t));
    }
}

static inline void
npu_flash_attention_store_i32_block(int32_t *dst, uint32_t dst_cols,
                                    uint32_t row0, uint32_t col0,
                                    uint32_t rows, uint32_t cols,
                                    const int32_t *src)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        memcpy(dst + (row0 + row) * dst_cols + col0, src + row * cols,
               cols * sizeof(int32_t));
    }
}

static inline void
npu_flash_attention_accumulate_i32_block(int32_t *dst, uint32_t dst_cols,
                                         uint32_t row0, uint32_t col0,
                                         uint32_t rows, uint32_t cols,
                                         const int32_t *src)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        int32_t *dst_row = dst + (row0 + row) * dst_cols + col0;
        const int32_t *src_row = src + row * cols;
        for (uint32_t col = 0U; col < cols; ++col) {
            dst_row[col] += src_row[col];
        }
    }
}

static inline int
npu_flash_attention_wait_i32_matrix_ready(
    const NpuMpuGemmSpmI32Matrix *matrix, uint32_t rows, uint32_t cols,
    uint64_t timeout)
{
    const volatile int32_t *spm =
        (const volatile int32_t *)(uintptr_t)matrix->base_addr;
    return npu_wait_i32_vector_not_value(NULL, spm, INT32_MIN, rows * cols,
                                         timeout);
}

static inline NpuMpuGemmSpmI8Matrix
npu_flash_attention_i8_matrix_view(uintptr_t base_addr, uint32_t rows,
                                   uint32_t cols)
{
    const NpuMpuGemmSpmI8Matrix matrix = {
        base_addr,
        rows,
        cols,
        npu_mpu_gemm_i8_row_stride_bytes(cols),
    };
    return matrix;
}

static inline NpuMpuGemmSpmI32Matrix
npu_flash_attention_i32_matrix_view(uintptr_t base_addr, uint32_t rows,
                                    uint32_t cols)
{
    const NpuMpuGemmSpmI32Matrix matrix = {
        base_addr,
        rows,
        cols,
        npu_mpu_gemm_i32_row_stride_bytes(cols),
    };
    return matrix;
}

static inline void
npu_flash_attention_init_online_softmax_state(
    const NpuFlashAttentionShape *shape, float *m_state, float *l_state)
{
    for (uint32_t row = 0U; row < shape->seq_q; ++row) {
        m_state[row] = -INFINITY;
        l_state[row] = 0.0f;
    }
}

static inline int
npu_flash_attention_run_single_gemm_tile(
    const NpuMpuGemmSpmI8Matrix *a_matrix,
    const NpuMpuGemmSpmI8Matrix *b_matrix,
    const NpuMpuGemmSpmI32Matrix *c_matrix, uint32_t m, uint32_t n,
    uint32_t k, const NpuMpuGemmLaunchConfig *launch, uint64_t wait_timeout,
    int32_t *tile_i32)
{
    const NpuMpuGemmProblemShape problem = {m, n, k};
    const NpuMpuGemmTileShape tile = {m, n, k};

    npu_mpu_gemm_fill_i32_matrix_in_spm(c_matrix, INT32_MIN);
    if (npu_mpu_gemm_launch_single_tile(a_matrix, b_matrix, c_matrix, &problem,
                                        &tile, launch, 0U, 0U) != 0) {
        return -1;
    }
    if (npu_flash_attention_wait_i32_matrix_ready(c_matrix, m, n,
                                                  wait_timeout) != 0) {
        return -1;
    }

    npu_mpu_gemm_load_i32_matrix_from_spm(c_matrix, tile_i32);
    return 0;
}

static inline int
npu_flash_attention_run_qk_stage(
    const NpuFlashAttentionShape *shape,
    const NpuFlashAttentionBuffers *buffers,
    const NpuFlashAttentionExecutionConfig *config)
{
    const NpuFlashAttentionQkShape qk_shape =
        npu_flash_attention_qk_shape_from_basic(shape);
    const NpuMpuGemmProblemShape problem =
        npu_flash_attention_qk_gemm_problem(&qk_shape);
    const uint32_t tile_m = config->qk_tile.tile_m;
    const uint32_t tile_n = config->qk_tile.tile_n;

    npu_flash_attention_qk_prepare_operands(
        buffers->q_f32, buffers->k_f32, &qk_shape, buffers->q_i8,
        buffers->q_metadata, buffers->k_i8, buffers->k_metadata,
        buffers->k_t_i8);

    for (uint32_t row0 = 0U; row0 < problem.m; row0 += tile_m) {
        const uint32_t rows =
            npu_mpu_gemm_tile_rows(&problem, &config->qk_tile, row0);
        for (uint32_t col0 = 0U; col0 < problem.n; col0 += tile_n) {
            const uint32_t cols =
                npu_mpu_gemm_tile_cols(&problem, &config->qk_tile, col0);
            const NpuMpuGemmSpmI8Matrix q_matrix =
                npu_flash_attention_i8_matrix_view(config->q_matrix.base_addr,
                                                   rows, problem.k);
            const NpuMpuGemmSpmI8Matrix k_t_matrix =
                npu_flash_attention_i8_matrix_view(
                    config->k_t_matrix.base_addr, problem.k, cols);
            const NpuMpuGemmSpmI32Matrix scores_matrix =
                npu_flash_attention_i32_matrix_view(
                    config->scores_matrix.base_addr, rows, cols);

            npu_mpu_gemm_store_i8_matrix_to_spm(
                buffers->q_i8 + row0 * problem.k, &q_matrix);
            npu_flash_attention_copy_i8_block(buffers->k_t_i8, problem.n, 0U,
                                              col0, problem.k, cols,
                                              buffers->k_t_tile_i8);
            npu_mpu_gemm_store_i8_matrix_to_spm(buffers->k_t_tile_i8,
                                                &k_t_matrix);

            if (npu_flash_attention_run_single_gemm_tile(
                    &q_matrix, &k_t_matrix, &scores_matrix, rows, cols,
                    problem.k, &config->qk_launch, config->wait_timeout,
                    buffers->tile_i32) != 0) {
                return NPU_FLASH_ATTENTION_QK_LAUNCH_FAILED;
            }
            npu_flash_attention_store_i32_block(
                buffers->scores_i32, problem.n, row0, col0, rows, cols,
                buffers->tile_i32);
        }
    }
    npu_flash_attention_qk_dequantize_scores_f32(
        buffers->scores_i32, buffers->q_metadata, buffers->k_metadata,
        &qk_shape, buffers->scores_f32);
    return NPU_FLASH_ATTENTION_OK;
}

static inline void
npu_flash_attention_run_online_softmax_stage(
    const NpuFlashAttentionShape *shape,
    const NpuFlashAttentionBuffers *buffers)
{
    npu_flash_attention_init_online_softmax_state(shape, buffers->m_state,
                                                  buffers->l_state);
    primitiveOnlineSoftmaxF32(
        buffers->scores_f32, shape->seq_q, shape->seq_k, buffers->m_state,
        buffers->l_state, buffers->m_state, buffers->l_state, buffers->p_f32);
}

static inline int
npu_flash_attention_run_pv_stage(
    const NpuFlashAttentionShape *shape,
    const NpuFlashAttentionBuffers *buffers,
    const NpuFlashAttentionExecutionConfig *config)
{
    const NpuFlashAttentionPvShape pv_shape =
        npu_flash_attention_pv_shape_from_basic(shape);
    const NpuMpuGemmProblemShape problem =
        npu_flash_attention_pv_gemm_problem(&pv_shape);
    const uint32_t tile_m = config->pv_tile.tile_m;
    const uint32_t tile_n = config->pv_tile.tile_n;
    const uint32_t tile_k = config->pv_tile.tile_k;

    npu_flash_attention_pv_prepare_operands(
        buffers->p_f32, buffers->v_f32, &pv_shape, buffers->p_i8,
        buffers->p_metadata, buffers->v_i8, buffers->v_metadata);

    for (uint32_t index = 0U; index < shape->seq_q * shape->dim_v; ++index) {
        buffers->out_i32[index] = 0;
    }

    for (uint32_t row0 = 0U; row0 < problem.m; row0 += tile_m) {
        const uint32_t rows =
            npu_mpu_gemm_tile_rows(&problem, &config->pv_tile, row0);
        for (uint32_t col0 = 0U; col0 < problem.n; col0 += tile_n) {
            const uint32_t cols =
                npu_mpu_gemm_tile_cols(&problem, &config->pv_tile, col0);
            for (uint32_t k0 = 0U; k0 < problem.k; k0 += tile_k) {
                const uint32_t chunk_k =
                    (problem.k - k0) < tile_k ? (problem.k - k0) : tile_k;
                const NpuMpuGemmSpmI8Matrix p_matrix =
                    npu_flash_attention_i8_matrix_view(
                        config->p_matrix.base_addr, rows, chunk_k);
                const NpuMpuGemmSpmI8Matrix v_matrix =
                    npu_flash_attention_i8_matrix_view(
                        config->v_matrix.base_addr, chunk_k, cols);
                const NpuMpuGemmSpmI32Matrix out_matrix =
                    npu_flash_attention_i32_matrix_view(
                        config->out_matrix.base_addr, rows, cols);

                npu_flash_attention_copy_i8_block(buffers->p_i8, problem.k,
                                                  row0, k0, rows, chunk_k,
                                                  buffers->p_tile_i8);
                npu_flash_attention_copy_i8_block(buffers->v_i8, problem.n,
                                                  k0, col0, chunk_k, cols,
                                                  buffers->v_tile_i8);
                npu_mpu_gemm_store_i8_matrix_to_spm(buffers->p_tile_i8,
                                                    &p_matrix);
                npu_mpu_gemm_store_i8_matrix_to_spm(buffers->v_tile_i8,
                                                    &v_matrix);

                if (npu_flash_attention_run_single_gemm_tile(
                        &p_matrix, &v_matrix, &out_matrix, rows, cols,
                        chunk_k, &config->pv_launch, config->wait_timeout,
                        buffers->tile_i32) != 0) {
                    return NPU_FLASH_ATTENTION_PV_LAUNCH_FAILED;
                }
                npu_flash_attention_accumulate_i32_block(
                    buffers->out_i32, problem.n, row0, col0, rows, cols,
                    buffers->tile_i32);
            }
        }
    }

    npu_flash_attention_pv_dequantize_outputs_f32(
        buffers->out_i32, buffers->p_metadata, buffers->v_metadata,
        &pv_shape, buffers->out_f32);
    return NPU_FLASH_ATTENTION_OK;
}

static inline int
npu_flash_attention_run_basic_f32(
    const NpuFlashAttentionShape *shape,
    const NpuFlashAttentionBuffers *buffers,
    const NpuFlashAttentionExecutionConfig *config)
{
    const int validate_status =
        npu_flash_attention_validate_basic_arguments(shape, buffers, config);
    int status = NPU_FLASH_ATTENTION_OK;
    if (validate_status != NPU_FLASH_ATTENTION_OK) {
        return validate_status;
    }

    status = npu_flash_attention_run_qk_stage(shape, buffers, config);
    if (status != NPU_FLASH_ATTENTION_OK) {
        return status;
    }

    npu_flash_attention_run_online_softmax_stage(shape, buffers);

    status = npu_flash_attention_run_pv_stage(shape, buffers, config);
    if (status != NPU_FLASH_ATTENTION_OK) {
        return status;
    }

    return NPU_FLASH_ATTENTION_OK;
}

#endif
