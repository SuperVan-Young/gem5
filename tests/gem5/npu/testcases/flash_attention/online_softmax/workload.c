/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <array>

#include "golden/llm_f32.hh"
#include "online_softmax.hh"

namespace
{

template <size_t Count>
static void
packFloatBits(const std::array<float, Count> &src, uint32_t *dst_bits)
{
    for (size_t index = 0U; index < Count; ++index) {
        __builtin_memcpy(&dst_bits[index], &src[index], sizeof(uint32_t));
    }
}

template <size_t Count>
static void
unpackFloatBits(const uint32_t *src_bits, std::array<float, Count> &dst)
{
    for (size_t index = 0U; index < Count; ++index) {
        __builtin_memcpy(&dst[index], &src_bits[index], sizeof(uint32_t));
    }
}

static bool
expectFloatClose(const char *label, float actual, float expected,
                 float tolerance)
{
    if (isnan(actual) || isnan(expected) ||
        fabsf(actual - expected) > tolerance) {
        printf("%s_FAIL actual=%.8f expected=%.8f\n", label, actual, expected);
        return false;
    }
    return true;
}

template <size_t Count>
static bool
expectArrayClose(const char *label, const std::array<float, Count> &actual,
                 const std::array<float, Count> &expected, float tolerance)
{
    for (size_t index = 0U; index < Count; ++index) {
        if (isnan(actual[index]) || isnan(expected[index]) ||
            fabsf(actual[index] - expected[index]) > tolerance) {
            printf("%s_FAIL[%u] actual=%.8f expected=%.8f\n", label,
                   (unsigned)index, actual[index], expected[index]);
            return false;
        }
    }
    return true;
}

template <size_t Cols>
static float
rowMax(const float *scores)
{
    float max_value = scores[0];
    for (size_t col = 1U; col < Cols; ++col) {
        if (scores[col] > max_value) {
            max_value = scores[col];
        }
    }
    return max_value;
}

template <size_t Cols>
static float
rowExpSum(const float *scores, float max_value)
{
    float sum = 0.0f;
    for (size_t col = 0U; col < Cols; ++col) {
        sum += expf(scores[col] - max_value);
    }
    return sum;
}

template <size_t Rows, size_t Cols>
static void
goldenSoftmax(const std::array<float, Rows * Cols> &scores,
              std::array<float, Rows * Cols> &prob)
{
    std::array<uint32_t, Rows * Cols> scores_bits = {};
    std::array<uint32_t, Rows * Cols> prob_bits = {};

    packFloatBits(scores, scores_bits.data());
    npu_golden_softmax_lastdim_f32(scores_bits.data(), Rows, Cols,
                                   prob_bits.data());
    unpackFloatBits(prob_bits.data(), prob);
}

} // namespace

int
main(void)
{
    static constexpr uint32_t Rows = 3U;
    static constexpr uint32_t SingleCols = 5U;
    static constexpr uint32_t Block0Cols = 2U;
    static constexpr uint32_t Block1Cols = 4U;
    static constexpr uint32_t TotalCols = Block0Cols + Block1Cols;
    static constexpr float Tolerance = 1.0e-5f;

    const std::array<float, Rows * SingleCols> single_scores = {
        0.25f, -0.50f, 1.50f, 0.00f, -1.25f,
        2.00f, 1.00f, -2.00f, 0.50f, -0.25f,
        -3.50f, -1.00f, -0.50f, -2.25f, -4.00f,
    };
    const std::array<float, Rows * TotalCols> full_scores = {
        1.00f, -1.50f, 0.25f, 3.00f, -2.00f, 1.50f,
        4.00f, 2.50f, 0.00f, -0.25f, -3.00f, 5.00f,
        -2.50f, -1.00f, -4.00f, -0.75f, 0.50f, -3.50f,
    };
    const std::array<float, Rows> init_m = {
        -INFINITY,
        -INFINITY,
        -INFINITY,
    };
    const std::array<float, Rows> init_l = {
        0.0f,
        0.0f,
        0.0f,
    };

    std::array<float, Rows * SingleCols> single_prob = {};
    std::array<float, Rows> single_m = {};
    std::array<float, Rows> single_l = {};
    std::array<float, Rows * SingleCols> single_expected = {};

    primitiveOnlineSoftmaxF32(single_scores.data(), Rows, SingleCols,
                              init_m.data(), init_l.data(), single_m.data(),
                              single_l.data(), single_prob.data());
    goldenSoftmax<Rows, SingleCols>(single_scores, single_expected);

    if (!expectArrayClose("FLASH_ATTENTION_ONLINE_SOFTMAX_SINGLE_BLOCK_PROB",
                          single_prob, single_expected, Tolerance)) {
        return 1;
    }
    for (uint32_t row = 0U; row < Rows; ++row) {
        const float expected_m = rowMax<SingleCols>(
            single_scores.data() + row * SingleCols);
        const float expected_l = rowExpSum<SingleCols>(
            single_scores.data() + row * SingleCols, expected_m);
        if (!expectFloatClose("FLASH_ATTENTION_ONLINE_SOFTMAX_SINGLE_BLOCK_M",
                              single_m[row], expected_m, Tolerance) ||
            !expectFloatClose("FLASH_ATTENTION_ONLINE_SOFTMAX_SINGLE_BLOCK_L",
                              single_l[row], expected_l, Tolerance)) {
            return 1;
        }
    }
    printf("FLASH_ATTENTION_ONLINE_SOFTMAX_SINGLE_BLOCK_PASS\n");

    std::array<float, Rows * Block0Cols> block0_scores = {};
    std::array<float, Rows * Block1Cols> block1_scores = {};
    for (uint32_t row = 0U; row < Rows; ++row) {
        for (uint32_t col = 0U; col < Block0Cols; ++col) {
            block0_scores[row * Block0Cols + col] =
                full_scores[row * TotalCols + col];
        }
        for (uint32_t col = 0U; col < Block1Cols; ++col) {
            block1_scores[row * Block1Cols + col] =
                full_scores[row * TotalCols + Block0Cols + col];
        }
    }

    std::array<float, Rows * Block0Cols> block0_prob = {};
    std::array<float, Rows * Block1Cols> block1_prob = {};
    std::array<float, Rows> block0_m = {};
    std::array<float, Rows> block0_l = {};
    std::array<float, Rows> final_m = {};
    std::array<float, Rows> final_l = {};
    std::array<float, Rows * TotalCols> online_full_prob = {};
    std::array<float, Rows * TotalCols> full_expected = {};

    primitiveOnlineSoftmaxF32(block0_scores.data(), Rows, Block0Cols,
                              init_m.data(), init_l.data(), block0_m.data(),
                              block0_l.data(), block0_prob.data());
    primitiveOnlineSoftmaxF32(block1_scores.data(), Rows, Block1Cols,
                              block0_m.data(), block0_l.data(), final_m.data(),
                              final_l.data(), block1_prob.data());
    goldenSoftmax<Rows, TotalCols>(full_scores, full_expected);

    for (uint32_t row = 0U; row < Rows; ++row) {
        const float block0_rescale = block0_l[row] == 0.0f ?
            0.0f :
            (block0_l[row] * expf(block0_m[row] - final_m[row]) /
             final_l[row]);
        for (uint32_t col = 0U; col < Block0Cols; ++col) {
            online_full_prob[row * TotalCols + col] =
                block0_prob[row * Block0Cols + col] * block0_rescale;
        }
        for (uint32_t col = 0U; col < Block1Cols; ++col) {
            online_full_prob[row * TotalCols + Block0Cols + col] =
                block1_prob[row * Block1Cols + col];
        }
    }

    if (!expectArrayClose("FLASH_ATTENTION_ONLINE_SOFTMAX_TWO_BLOCK_PROB",
                          online_full_prob, full_expected, Tolerance)) {
        return 1;
    }
    for (uint32_t row = 0U; row < Rows; ++row) {
        const float expected_m =
            rowMax<TotalCols>(full_scores.data() + row * TotalCols);
        const float expected_l = rowExpSum<TotalCols>(
            full_scores.data() + row * TotalCols, expected_m);
        float row_sum = 0.0f;
        for (uint32_t col = 0U; col < TotalCols; ++col) {
            row_sum += online_full_prob[row * TotalCols + col];
        }
        if (!expectFloatClose("FLASH_ATTENTION_ONLINE_SOFTMAX_TWO_BLOCK_M",
                              final_m[row], expected_m, Tolerance) ||
            !expectFloatClose("FLASH_ATTENTION_ONLINE_SOFTMAX_TWO_BLOCK_L",
                              final_l[row], expected_l, Tolerance) ||
            !expectFloatClose("FLASH_ATTENTION_ONLINE_SOFTMAX_TWO_BLOCK_SUM",
                              row_sum, 1.0f, Tolerance)) {
            return 1;
        }
    }
    printf("FLASH_ATTENTION_ONLINE_SOFTMAX_TWO_BLOCK_PASS\n");

    printf("FLASH_ATTENTION_ONLINE_SOFTMAX_PASS\n");
    return 0;
}
