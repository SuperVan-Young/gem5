#ifndef TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_HH_
#define TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_HH_

#include <cmath>
#include <cstdint>

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
        buffers->v_f32 == NULL || buffers->expected_scores_i32 == NULL ||
        buffers->expected_out_i32 == NULL || buffers->q_i8 == NULL ||
        buffers->k_i8 == NULL || buffers->k_t_i8 == NULL ||
        buffers->v_i8 == NULL || buffers->scores_i32 == NULL ||
        buffers->scores_f32 == NULL || buffers->m_state == NULL ||
        buffers->l_state == NULL || buffers->p_f32 == NULL ||
        buffers->p_i8 == NULL || buffers->out_i32 == NULL ||
        buffers->out_f32 == NULL || buffers->q_metadata == NULL ||
        buffers->k_metadata == NULL || buffers->p_metadata == NULL ||
        buffers->v_metadata == NULL) {
        return NPU_FLASH_ATTENTION_INVALID_ARGUMENT;
    }
    if (config->wait_timeout == 0ULL) {
        return NPU_FLASH_ATTENTION_INVALID_ARGUMENT;
    }

    return NPU_FLASH_ATTENTION_OK;
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
npu_flash_attention_run_qk_stage(
    const NpuFlashAttentionShape *shape,
    const NpuFlashAttentionBuffers *buffers,
    const NpuFlashAttentionExecutionConfig *config)
{
    const NpuFlashAttentionQkShape qk_shape =
        npu_flash_attention_qk_shape_from_basic(shape);
    const NpuMpuGemmProblemShape problem =
        npu_flash_attention_qk_gemm_problem(&qk_shape);

    npu_flash_attention_qk_prepare_operands(
        buffers->q_f32, buffers->k_f32, &qk_shape, buffers->q_i8,
        buffers->q_metadata, buffers->k_i8, buffers->k_metadata,
        buffers->k_t_i8);

    npu_mpu_gemm_store_i8_matrix_to_spm(buffers->q_i8, &config->q_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(buffers->k_t_i8,
                                        &config->k_t_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&config->scores_matrix, -1);

    if (npu_mpu_gemm_launch_tiled_mn(
            &config->q_matrix, &config->k_t_matrix,
            &config->scores_matrix, &problem, &config->qk_tile,
            &config->qk_launch) != 0) {
        return NPU_FLASH_ATTENTION_QK_LAUNCH_FAILED;
    }
    if (npu_wait_u32_vector_match(
            NULL,
            (volatile uint32_t *)(uintptr_t)config->scores_matrix.base_addr,
            (const uint32_t *)buffers->expected_scores_i32,
            shape->seq_q * shape->seq_k, config->wait_timeout) != 0) {
        return NPU_FLASH_ATTENTION_QK_WAIT_FAILED;
    }

    npu_mpu_gemm_load_i32_matrix_from_spm(&config->scores_matrix,
                                          buffers->scores_i32);
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

    npu_flash_attention_pv_prepare_operands(
        buffers->p_f32, buffers->v_f32, &pv_shape, buffers->p_i8,
        buffers->p_metadata, buffers->v_i8, buffers->v_metadata);

    npu_mpu_gemm_store_i8_matrix_to_spm(buffers->p_i8, &config->p_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(buffers->v_i8, &config->v_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&config->out_matrix, -1);

    if (npu_mpu_gemm_launch_tiled_mn(
            &config->p_matrix, &config->v_matrix, &config->out_matrix,
            &problem, &config->pv_tile, &config->pv_launch) != 0) {
        return NPU_FLASH_ATTENTION_PV_LAUNCH_FAILED;
    }
    if (npu_wait_u32_vector_match(
            NULL,
            (volatile uint32_t *)(uintptr_t)config->out_matrix.base_addr,
            (const uint32_t *)buffers->expected_out_i32,
            shape->seq_q * shape->dim_v,
            config->wait_timeout) != 0) {
        return NPU_FLASH_ATTENTION_PV_WAIT_FAILED;
    }

    npu_mpu_gemm_load_i32_matrix_from_spm(&config->out_matrix,
                                          buffers->out_i32);
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
        npu_mpu_gemm_load_i32_matrix_from_spm(&config->scores_matrix,
                                              buffers->scores_i32);
        return status;
    }

    npu_flash_attention_run_online_softmax_stage(shape, buffers);

    status = npu_flash_attention_run_pv_stage(shape, buffers, config);
    if (status != NPU_FLASH_ATTENTION_OK) {
        npu_mpu_gemm_load_i32_matrix_from_spm(&config->out_matrix,
                                              buffers->out_i32);
        return status;
    }

    return NPU_FLASH_ATTENTION_OK;
}

#endif
