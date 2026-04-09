#include "llm_f32.hh"

#include <math.h>

namespace
{

static inline float
bitsToFloat(uint32_t bits)
{
    float value = 0.0f;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline uint32_t
floatToBits(float value)
{
    uint32_t bits = 0U;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

} // namespace

void
npu_golden_rmsnorm_lastdim_f32(const uint32_t *src_bits,
                               const uint32_t *weight_bits, uint32_t rows,
                               uint32_t cols, float epsilon, uint32_t *dst_bits)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        float mean_square = 0.0f;
        for (uint32_t col = 0U; col < cols; ++col) {
            const float value = bitsToFloat(src_bits[row * cols + col]);
            mean_square += value * value;
        }
        mean_square /= static_cast<float>(cols);
        const float inv_rms = 1.0f / sqrtf(mean_square + epsilon);
        for (uint32_t col = 0U; col < cols; ++col) {
            const float value = bitsToFloat(src_bits[row * cols + col]);
            const float weight = bitsToFloat(weight_bits[col]);
            dst_bits[row * cols + col] = floatToBits(value * inv_rms * weight);
        }
    }
}

void
npu_golden_softmax_lastdim_f32(const uint32_t *src_bits, uint32_t rows,
                               uint32_t cols, uint32_t *dst_bits)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        float max_value = bitsToFloat(src_bits[row * cols]);
        float sum = 0.0f;
        for (uint32_t col = 1U; col < cols; ++col) {
            const float value = bitsToFloat(src_bits[row * cols + col]);
            if (value > max_value) {
                max_value = value;
            }
        }
        for (uint32_t col = 0U; col < cols; ++col) {
            const float exp_value =
                expf(bitsToFloat(src_bits[row * cols + col]) - max_value);
            dst_bits[row * cols + col] = floatToBits(exp_value);
            sum += exp_value;
        }
        for (uint32_t col = 0U; col < cols; ++col) {
            dst_bits[row * cols + col] = floatToBits(
                bitsToFloat(dst_bits[row * cols + col]) / sum);
        }
    }
}

void
npu_golden_swiglu_lastdim_f32(const uint32_t *gate_bits,
                              const uint32_t *value_bits, uint32_t rows,
                              uint32_t cols, uint32_t *dst_bits)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        for (uint32_t col = 0U; col < cols; ++col) {
            const float gate = bitsToFloat(gate_bits[row * cols + col]);
            const float value = bitsToFloat(value_bits[row * cols + col]);
            const float sigmoid = 1.0f / (1.0f + expf(-gate));
            dst_bits[row * cols + col] = floatToBits(gate * sigmoid * value);
        }
    }
}
