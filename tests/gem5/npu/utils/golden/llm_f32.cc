#include <cmath>

#include "llm_f32.hh"

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

float
npu_golden_flashattention_default_scale_f32(uint32_t dim)
{
    if (dim == 0U) {
        return 0.0f;
    }

    return 1.0f / sqrtf(static_cast<float>(dim));
}

void
npu_golden_rmsnorm_lastdim_f32(const uint32_t *src_bits,
                               const uint32_t *weight_bits, uint32_t rows,
                               uint32_t cols, float epsilon,
                               uint32_t *dst_bits)
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

void
npu_golden_qk_scores_f32(const uint32_t *q_bits, const uint32_t *k_bits,
                         uint32_t seq_q, uint32_t seq_k, uint32_t dim,
                         float scaling, uint32_t *scores_bits)
{
    if (q_bits == nullptr || k_bits == nullptr || scores_bits == nullptr ||
        seq_q == 0U || seq_k == 0U) {
        return;
    }

    if (dim == 0U) {
        for (uint32_t index = 0U; index < seq_q * seq_k; ++index) {
            scores_bits[index] = floatToBits(0.0f);
        }
        return;
    }

    for (uint32_t row = 0U; row < seq_q; ++row) {
        for (uint32_t col = 0U; col < seq_k; ++col) {
            float dot = 0.0f;
            for (uint32_t d = 0U; d < dim; ++d) {
                const float q_value = bitsToFloat(q_bits[row * dim + d]);
                const float k_value = bitsToFloat(k_bits[col * dim + d]);
                dot += q_value * k_value;
            }
            scores_bits[row * seq_k + col] = floatToBits(dot * scaling);
        }
    }
}

void
npu_golden_attention_probs_f32(const uint32_t *scores_bits, uint32_t seq_q,
                               uint32_t seq_k, uint32_t *prob_bits)
{
    if (scores_bits == nullptr || prob_bits == nullptr || seq_q == 0U ||
        seq_k == 0U) {
        return;
    }

    npu_golden_softmax_lastdim_f32(scores_bits, seq_q, seq_k, prob_bits);
}

void
npu_golden_flashattention_f32(const uint32_t *q_bits, const uint32_t *k_bits,
                              const uint32_t *v_bits, uint32_t seq_q,
                              uint32_t seq_k, uint32_t dim, uint32_t dim_v,
                              float scaling, uint32_t *scores_bits,
                              uint32_t *prob_bits, uint32_t *dst_bits)
{
    if (q_bits == nullptr || k_bits == nullptr || v_bits == nullptr ||
        scores_bits == nullptr || prob_bits == nullptr ||
        dst_bits == nullptr || seq_q == 0U || seq_k == 0U || dim_v == 0U) {
        return;
    }

    npu_golden_qk_scores_f32(q_bits, k_bits, seq_q, seq_k, dim, scaling,
                             scores_bits);
    npu_golden_attention_probs_f32(scores_bits, seq_q, seq_k, prob_bits);

    for (uint32_t row = 0U; row < seq_q; ++row) {
        for (uint32_t col = 0U; col < dim_v; ++col) {
            float acc = 0.0f;
            for (uint32_t key = 0U; key < seq_k; ++key) {
                const float prob = bitsToFloat(prob_bits[row * seq_k + key]);
                const float value = bitsToFloat(v_bits[key * dim_v + col]);
                acc += prob * value;
            }
            dst_bits[row * dim_v + col] = floatToBits(acc);
        }
    }
}
