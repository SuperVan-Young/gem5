#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_QUANTIZE_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_QUANTIZE_HH_

#include <cmath>
#include <cstddef>
#include <cstdint>

struct PrimitiveScaleMetadata
{
    float inputToIntScale = 0.0f;
    float intToInputScale = 0.0f;
    float absMax = 0.0f;
};

static inline float
primitiveQuantAbsF32(float value)
{
    return value < 0.0f ? -value : value;
}

static inline int32_t
primitiveQuantClampI32(int32_t value, int32_t lo, int32_t hi)
{
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static inline PrimitiveScaleMetadata
primitiveQuantChooseSymmetricI8ScaleF32(const float *src, uint32_t count)
{
    PrimitiveScaleMetadata meta = {};
    for (uint32_t i = 0U; i < count; ++i) {
        const float abs_value = primitiveQuantAbsF32(src[i]);
        if (abs_value > meta.absMax) {
            meta.absMax = abs_value;
        }
    }

    if (meta.absMax == 0.0f) {
        meta.inputToIntScale = 0.0f;
        meta.intToInputScale = 0.0f;
        return meta;
    }

    meta.inputToIntScale = 127.0f / meta.absMax;
    meta.intToInputScale = meta.absMax / 127.0f;
    return meta;
}

static inline void
primitiveQuantizeRowwiseF32ToI8(const float *src, uint32_t rows, uint32_t cols,
                                int8_t *dst,
                                PrimitiveScaleMetadata *row_metadata)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        const PrimitiveScaleMetadata meta =
            primitiveQuantChooseSymmetricI8ScaleF32(src + row * cols, cols);
        if (row_metadata != nullptr) {
            row_metadata[row] = meta;
        }

        for (uint32_t col = 0U; col < cols; ++col) {
            const float scaled = src[row * cols + col] * meta.inputToIntScale;
            const int32_t quantized = primitiveQuantClampI32(
                static_cast<int32_t>(std::lrintf(scaled)), -127, 127);
            dst[row * cols + col] = static_cast<int8_t>(quantized);
        }
    }
}

static inline PrimitiveScaleMetadata
primitiveDequantizeI32ToF32(const int32_t *src, uint32_t count,
                            float int_to_input_scale, float *dst)
{
    PrimitiveScaleMetadata meta = {};
    meta.intToInputScale = int_to_input_scale;
    meta.inputToIntScale =
        int_to_input_scale == 0.0f ? 0.0f : 1.0f / int_to_input_scale;

    for (uint32_t i = 0U; i < count; ++i) {
        const float value =
            static_cast<float>(src[i]) * meta.intToInputScale;
        dst[i] = value;
        const float abs_value = primitiveQuantAbsF32(value);
        if (abs_value > meta.absMax) {
            meta.absMax = abs_value;
        }
    }

    return meta;
}

#endif
