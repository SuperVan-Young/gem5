#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_ONLINE_SOFTMAX_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_ONLINE_SOFTMAX_HH_

#include <cmath>
#include <cstdint>

static inline void
primitiveOnlineSoftmaxF32(const float *scores_block, uint32_t rows,
                          uint32_t cols, const float *m_prev,
                          const float *l_prev, float *m_next, float *l_next,
                          float *p_block)
{
    if (scores_block == nullptr || m_prev == nullptr || l_prev == nullptr ||
        m_next == nullptr || l_next == nullptr || p_block == nullptr ||
        rows == 0U || cols == 0U) {
        return;
    }

    for (uint32_t row = 0U; row < rows; ++row) {
        const uint32_t row_offset = row * cols;
        float block_max = scores_block[row_offset];
        for (uint32_t col = 1U; col < cols; ++col) {
            const float value = scores_block[row_offset + col];
            if (value > block_max) {
                block_max = value;
            }
        }

        const float next_max =
            m_prev[row] > block_max ? m_prev[row] : block_max;
        m_next[row] = next_max;

        if (!std::isfinite(next_max)) {
            l_next[row] = 0.0f;
            for (uint32_t col = 0U; col < cols; ++col) {
                p_block[row_offset + col] = 0.0f;
            }
            continue;
        }

        const float prev_scale = l_prev[row] == 0.0f ?
            0.0f :
            l_prev[row] * expf(m_prev[row] - next_max);
        float block_sum = 0.0f;
        for (uint32_t col = 0U; col < cols; ++col) {
            const float exp_value =
                expf(scores_block[row_offset + col] - next_max);
            p_block[row_offset + col] = exp_value;
            block_sum += exp_value;
        }

        const float next_l = prev_scale + block_sum;
        l_next[row] = next_l;
        if (next_l == 0.0f) {
            for (uint32_t col = 0U; col < cols; ++col) {
                p_block[row_offset + col] = 0.0f;
            }
            continue;
        }

        for (uint32_t col = 0U; col < cols; ++col) {
            p_block[row_offset + col] /= next_l;
        }
    }
}

#endif
