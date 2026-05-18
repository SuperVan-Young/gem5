#ifndef TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_QK_HH_
#define TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_QK_HH_

#include <cmath>
#include <cstdint>

#include "mpu_gemm_layout.hh"
#include "primitive/quantize.hh"

typedef struct
{
    uint32_t seq_q;
    uint32_t seq_k;
    uint32_t dim;
} NpuFlashAttentionQkShape;

static inline float
npu_flash_attention_qk_scale_f32(uint32_t dim)
{
    if (dim == 0U) {
        return 0.0f;
    }

    return 1.0f / sqrtf((float)dim);
}

static inline void
npu_flash_attention_qk_quantize_inputs(
    const float *q_f32, const float *k_f32,
    const NpuFlashAttentionQkShape *shape, int8_t *q_i8,
    PrimitiveScaleMetadata *q_metadata, int8_t *k_i8,
    PrimitiveScaleMetadata *k_metadata)
{
    primitiveQuantizeRowwiseF32ToI8(q_f32, shape->seq_q, shape->dim, q_i8,
                                    q_metadata);
    primitiveQuantizeRowwiseF32ToI8(k_f32, shape->seq_k, shape->dim, k_i8,
                                    k_metadata);
}

static inline void
npu_flash_attention_qk_materialize_k_transpose_i8(
    const int8_t *k_i8, const NpuFlashAttentionQkShape *shape, int8_t *k_t_i8)
{
    for (uint32_t row = 0U; row < shape->seq_k; ++row) {
        for (uint32_t col = 0U; col < shape->dim; ++col) {
            k_t_i8[col * shape->seq_k + row] = k_i8[row * shape->dim + col];
        }
    }
}

static inline void
npu_flash_attention_qk_prepare_operands(
    const float *q_f32, const float *k_f32,
    const NpuFlashAttentionQkShape *shape, int8_t *q_i8,
    PrimitiveScaleMetadata *q_metadata, int8_t *k_i8,
    PrimitiveScaleMetadata *k_metadata, int8_t *k_t_i8)
{
    npu_flash_attention_qk_quantize_inputs(q_f32, k_f32, shape, q_i8,
                                           q_metadata, k_i8, k_metadata);
    npu_flash_attention_qk_materialize_k_transpose_i8(k_i8, shape, k_t_i8);
}

static inline NpuMpuGemmProblemShape
npu_flash_attention_qk_gemm_problem(
    const NpuFlashAttentionQkShape *shape)
{
    const NpuMpuGemmProblemShape problem = {
        shape->seq_q,
        shape->seq_k,
        shape->dim,
    };
    return problem;
}

static inline float
npu_flash_attention_qk_dequantize_score_f32(
    int32_t score_i32, const PrimitiveScaleMetadata *q_metadata,
    const PrimitiveScaleMetadata *k_metadata, uint32_t dim)
{
    float score_f32 = 0.0f;

    primitiveDequantizeI32ToF32(
        &score_i32, 1U,
        q_metadata->intToInputScale * k_metadata->intToInputScale,
        &score_f32);
    return score_f32 * npu_flash_attention_qk_scale_f32(dim);
}

static inline void
npu_flash_attention_qk_dequantize_scores_f32(
    const int32_t *scores_i32, const PrimitiveScaleMetadata *q_metadata,
    const PrimitiveScaleMetadata *k_metadata,
    const NpuFlashAttentionQkShape *shape, float *scores_f32)
{
    for (uint32_t row = 0U; row < shape->seq_q; ++row) {
        for (uint32_t col = 0U; col < shape->seq_k; ++col) {
            scores_f32[row * shape->seq_k + col] =
                npu_flash_attention_qk_dequantize_score_f32(
                    scores_i32[row * shape->seq_k + col], &q_metadata[row],
                    &k_metadata[col], shape->dim);
        }
    }
}

#endif
