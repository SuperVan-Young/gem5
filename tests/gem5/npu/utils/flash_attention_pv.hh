#ifndef TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_PV_HH_
#define TESTS_GEM5_NPU_UTILS_FLASH_ATTENTION_PV_HH_

#include <cmath>
#include <cstdint>

#include "mpu_gemm_layout.hh"
#include "primitive/quantize.hh"

typedef struct
{
    uint32_t seq_q;
    uint32_t seq_k;
    uint32_t dim_v;
} NpuFlashAttentionPvShape;

static inline void
npu_flash_attention_pv_quantize_probs(
    const float *p_f32, const NpuFlashAttentionPvShape *shape, int8_t *p_i8,
    PrimitiveScaleMetadata *p_metadata)
{
    primitiveQuantizeRowwiseF32ToI8(p_f32, shape->seq_q, shape->seq_k, p_i8,
                                    p_metadata);
}

static inline PrimitiveScaleMetadata
npu_flash_attention_pv_choose_column_scale_f32(const float *v_f32,
                                               uint32_t rows, uint32_t cols,
                                               uint32_t col)
{
    PrimitiveScaleMetadata meta = {};

    for (uint32_t row = 0U; row < rows; ++row) {
        const float abs_value =
            primitiveQuantAbsF32(v_f32[row * cols + col]);
        if (abs_value > meta.absMax) {
            meta.absMax = abs_value;
        }
    }

    if (meta.absMax == 0.0f) {
        return meta;
    }

    meta.inputToIntScale = 127.0f / meta.absMax;
    meta.intToInputScale = meta.absMax / 127.0f;
    return meta;
}

static inline void
npu_flash_attention_pv_quantize_values(
    const float *v_f32, const NpuFlashAttentionPvShape *shape, int8_t *v_i8,
    PrimitiveScaleMetadata *v_metadata)
{
    for (uint32_t col = 0U; col < shape->dim_v; ++col) {
        const PrimitiveScaleMetadata meta =
            npu_flash_attention_pv_choose_column_scale_f32(
                v_f32, shape->seq_k, shape->dim_v, col);

        if (v_metadata != nullptr) {
            v_metadata[col] = meta;
        }

        for (uint32_t row = 0U; row < shape->seq_k; ++row) {
            const float scaled =
                v_f32[row * shape->dim_v + col] * meta.inputToIntScale;
            const int32_t quantized = primitiveQuantClampI32(
                static_cast<int32_t>(std::lrintf(scaled)), -127, 127);
            v_i8[row * shape->dim_v + col] = static_cast<int8_t>(quantized);
        }
    }
}

static inline void
npu_flash_attention_pv_prepare_operands(
    const float *p_f32, const float *v_f32,
    const NpuFlashAttentionPvShape *shape, int8_t *p_i8,
    PrimitiveScaleMetadata *p_metadata, int8_t *v_i8,
    PrimitiveScaleMetadata *v_metadata)
{
    npu_flash_attention_pv_quantize_probs(p_f32, shape, p_i8, p_metadata);
    npu_flash_attention_pv_quantize_values(v_f32, shape, v_i8, v_metadata);
}

static inline NpuMpuGemmProblemShape
npu_flash_attention_pv_gemm_problem(
    const NpuFlashAttentionPvShape *shape)
{
    const NpuMpuGemmProblemShape problem = {
        shape->seq_q,
        shape->dim_v,
        shape->seq_k,
    };
    return problem;
}

static inline float
npu_flash_attention_pv_dequantize_out_f32(
    int32_t out_i32, const PrimitiveScaleMetadata *p_metadata,
    const PrimitiveScaleMetadata *v_metadata)
{
    float out_f32 = 0.0f;

    primitiveDequantizeI32ToF32(
        &out_i32, 1U,
        p_metadata->intToInputScale * v_metadata->intToInputScale, &out_f32);
    return out_f32;
}

static inline void
npu_flash_attention_pv_dequantize_outputs_f32(
    const int32_t *out_i32, const PrimitiveScaleMetadata *p_metadata,
    const PrimitiveScaleMetadata *v_metadata,
    const NpuFlashAttentionPvShape *shape, float *out_f32)
{
    for (uint32_t row = 0U; row < shape->seq_q; ++row) {
        for (uint32_t col = 0U; col < shape->dim_v; ++col) {
            out_f32[row * shape->dim_v + col] =
                npu_flash_attention_pv_dequantize_out_f32(
                    out_i32[row * shape->dim_v + col], &p_metadata[row],
                    &v_metadata[col]);
        }
    }
}

#endif
