/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <array>

#include "golden/llm_f32.hh"

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

static bool
expectClose(const char *label, const uint32_t *actual_bits,
            const float *expected_values, size_t count, float tolerance)
{
    for (size_t index = 0; index < count; ++index) {
        const float actual = bitsToFloat(actual_bits[index]);
        const float expected = expected_values[index];
        if (isnan(actual) || fabsf(actual - expected) > tolerance) {
            printf("%s_MISMATCH[%u]=%.8f expected=%.8f\n", label,
                   (unsigned)index, actual, expected);
            return false;
        }
    }
    return true;
}

static bool
expectBitwiseClose(const char *label, const uint32_t *lhs_bits,
                   const uint32_t *rhs_bits, size_t count, float tolerance)
{
    for (size_t index = 0; index < count; ++index) {
        const float lhs = bitsToFloat(lhs_bits[index]);
        const float rhs = bitsToFloat(rhs_bits[index]);
        if (isnan(lhs) || isnan(rhs) || fabsf(lhs - rhs) > tolerance) {
            printf("%s_MISMATCH[%u]=%.8f rhs=%.8f\n", label, (unsigned)index,
                   lhs, rhs);
            return false;
        }
    }
    return true;
}

static bool
expectProbRowsSumToOne(const uint32_t *prob_bits, uint32_t rows, uint32_t cols,
                       float tolerance)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0U; col < cols; ++col) {
            sum += bitsToFloat(prob_bits[row * cols + col]);
        }
        if (fabsf(sum - 1.0f) > tolerance) {
            printf("FLASH_ATTENTION_GOLDEN_PROB_SUM_FAIL[%u]=%.8f\n",
                   (unsigned)row, sum);
            return false;
        }
    }
    return true;
}

static void
printVector(const char *label, const uint32_t *values, size_t count)
{
    printf("%s=", label);
    for (size_t index = 0; index < count; ++index) {
        if (index != 0U) {
            putchar(',');
        }
        printf("%.6f", (double)bitsToFloat(values[index]));
    }
    putchar('\n');
}

} // namespace

int
main(void)
{
    static constexpr uint32_t SeqQ = 2U;
    static constexpr uint32_t SeqK = 3U;
    static constexpr uint32_t Dim = 2U;
    static constexpr uint32_t DimV = 2U;
    static constexpr float Tolerance = 1.0e-5f;

    const float scaling = npu_golden_flashattention_default_scale_f32(Dim);
    const float expected_scaling = 1.0f / sqrtf(static_cast<float>(Dim));
    if (fabsf(scaling - expected_scaling) > Tolerance) {
        printf("FLASH_ATTENTION_GOLDEN_SCALE_FAIL=%.8f expected=%.8f\n",
               scaling, expected_scaling);
        return 1;
    }

    const std::array<uint32_t, SeqQ * Dim> q = {
        floatToBits(1.0f),
        floatToBits(0.0f),
        floatToBits(0.0f),
        floatToBits(1.0f),
    };
    const std::array<uint32_t, SeqK * Dim> k = {
        floatToBits(1.0f),
        floatToBits(0.0f),
        floatToBits(0.0f),
        floatToBits(1.0f),
        floatToBits(1.0f),
        floatToBits(1.0f),
    };
    const std::array<uint32_t, SeqK * DimV> v = {
        floatToBits(1.0f),
        floatToBits(2.0f),
        floatToBits(3.0f),
        floatToBits(4.0f),
        floatToBits(5.0f),
        floatToBits(6.0f),
    };

    std::array<float, SeqQ * SeqK> expected_scores = {
        scaling,
        0.0f,
        scaling,
        0.0f,
        scaling,
        scaling,
    };

    const float exp_scaled = expf(scaling);
    const float denom = 2.0f * exp_scaled + 1.0f;
    const float p_hi = exp_scaled / denom;
    const float p_lo = 1.0f / denom;
    std::array<float, SeqQ * SeqK> expected_probs = {
        p_hi,
        p_lo,
        p_hi,
        p_lo,
        p_hi,
        p_hi,
    };
    std::array<float, SeqQ * DimV> expected_output = {
        p_hi * 1.0f + p_lo * 3.0f + p_hi * 5.0f,
        p_hi * 2.0f + p_lo * 4.0f + p_hi * 6.0f,
        p_lo * 1.0f + p_hi * 3.0f + p_hi * 5.0f,
        p_lo * 2.0f + p_hi * 4.0f + p_hi * 6.0f,
    };

    std::array<uint32_t, SeqQ * SeqK> staged_scores = {};
    std::array<uint32_t, SeqQ * SeqK> staged_probs = {};
    std::array<uint32_t, SeqQ * SeqK> flash_scores = {};
    std::array<uint32_t, SeqQ * SeqK> flash_probs = {};
    std::array<uint32_t, SeqQ * DimV> flash_output = {};

    npu_golden_qk_scores_f32(q.data(), k.data(), SeqQ, SeqK, Dim, scaling,
                             staged_scores.data());
    npu_golden_attention_probs_f32(staged_scores.data(), SeqQ, SeqK,
                                   staged_probs.data());
    npu_golden_flashattention_f32(
        q.data(), k.data(), v.data(), SeqQ, SeqK, Dim, DimV, scaling,
        flash_scores.data(), flash_probs.data(), flash_output.data());

    printVector("FLASH_ATTENTION_GOLDEN_SCORES", flash_scores.data(),
                flash_scores.size());
    printVector("FLASH_ATTENTION_GOLDEN_PROBS", flash_probs.data(),
                flash_probs.size());
    printVector("FLASH_ATTENTION_GOLDEN_OUTPUT", flash_output.data(),
                flash_output.size());

    if (!expectClose("FLASH_ATTENTION_GOLDEN_SCORES", staged_scores.data(),
                     expected_scores.data(), staged_scores.size(),
                     Tolerance)) {
        return 1;
    }
    if (!expectBitwiseClose("FLASH_ATTENTION_GOLDEN_SCORES_STAGE_COMPARE",
                            staged_scores.data(), flash_scores.data(),
                            staged_scores.size(), Tolerance)) {
        return 1;
    }
    if (!expectClose("FLASH_ATTENTION_GOLDEN_PROBS", flash_probs.data(),
                     expected_probs.data(), flash_probs.size(), Tolerance)) {
        return 1;
    }
    if (!expectBitwiseClose("FLASH_ATTENTION_GOLDEN_PROBS_STAGE_COMPARE",
                            staged_probs.data(), flash_probs.data(),
                            staged_probs.size(), Tolerance)) {
        return 1;
    }
    if (!expectProbRowsSumToOne(flash_probs.data(), SeqQ, SeqK, Tolerance)) {
        return 1;
    }
    if (!expectClose("FLASH_ATTENTION_GOLDEN_OUTPUT", flash_output.data(),
                     expected_output.data(), flash_output.size(), Tolerance)) {
        return 1;
    }

    printf("FLASH_ATTENTION_GOLDEN_CONTRACT_PASS\n");
    return 0;
}
