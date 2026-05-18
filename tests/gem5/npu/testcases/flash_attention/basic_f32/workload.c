/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <array>

#include "flash_attention.hh"
#include "golden/llm_f32.hh"
#include "golden/mpu_gemm.hh"
#include "npu_assert.hh"

namespace
{

static constexpr uint32_t SeqQ = 2U;
static constexpr uint32_t SeqK = 3U;
static constexpr uint32_t Dim = 2U;
static constexpr uint32_t DimV = 2U;
static constexpr uint32_t QkTileM = 2U;
static constexpr uint32_t QkTileN = 2U;
static constexpr uint32_t QkTileK = Dim;
static constexpr uint32_t PvTileM = 2U;
static constexpr uint32_t PvTileN = 1U;
static constexpr uint32_t PvTileK = SeqK;
static constexpr uintptr_t QSpm = 0x60008000ULL;
static constexpr uintptr_t KTSpm = 0x60009000ULL;
static constexpr uintptr_t ScoresSpm = 0x6000a000ULL;
static constexpr uintptr_t PSpm = 0x6000b000ULL;
static constexpr uintptr_t VSpm = 0x6000c000ULL;
static constexpr uintptr_t OutSpm = 0x6000d000ULL;
static constexpr uint32_t DeviceId = 0U;
static constexpr uint32_t SyncIndicator = 0U;
static constexpr uint32_t SetCompletionSync = 0U;
static constexpr uint64_t WaitTimeout = 80000000ULL;
static constexpr float AbsTolerance = 1.0e-6f;
static constexpr float RelTolerance = 1.0e-6f;

static void
packFloatBits(const float *src, uint32_t count, uint32_t *dst_bits)
{
    for (uint32_t index = 0U; index < count; ++index) {
        dst_bits[index] = npu_float_to_bits(src[index]);
    }
}

static void
unpackFloatBits(const uint32_t *src_bits, uint32_t count, float *dst)
{
    for (uint32_t index = 0U; index < count; ++index) {
        dst[index] = npu_bits_to_float(src_bits[index]);
    }
}

static void
printI32Vector(const char *label, const int32_t *values, uint32_t count)
{
    printf("%s=", label);
    for (uint32_t index = 0U; index < count; ++index) {
        if (index != 0U) {
            putchar(',');
        }
        printf("%d", values[index]);
    }
    putchar('\n');
}

static void
printF32Vector(const char *label, const float *values, uint32_t count)
{
    printf("%s=", label);
    for (uint32_t index = 0U; index < count; ++index) {
        if (index != 0U) {
            putchar(',');
        }
        printf("%.6f", (double)values[index]);
    }
    putchar('\n');
}

static int
expectStatus(const char *label, int actual, int expected)
{
    if (actual == expected) {
        return 0;
    }

    printf("%s actual=%d expected=%d\n", label, actual, expected);
    return -1;
}

} // namespace

int
main(int argc, char **argv)
{
    const char *scenario = argc > 1 ? argv[1] : "flash_attention_basic_f32";
    const std::array<float, SeqQ * Dim> q_f32 = {
        127.0f, 0.0f,
        0.0f, 127.0f,
    };
    const std::array<float, SeqK * Dim> k_f32 = {
        127.0f, 0.0f,
        0.0f, 127.0f,
        127.0f, 127.0f,
    };
    const std::array<float, SeqK * DimV> v_f32 = {
        127.0f, 64.0f,
        63.0f, 127.0f,
        -127.0f, -64.0f,
    };
    const NpuFlashAttentionShape shape = {
        SeqQ,
        SeqK,
        Dim,
        DimV,
    };
    const NpuFlashAttentionExecutionConfig exec = {
        {
            QSpm,
            SeqQ,
            Dim,
            npu_mpu_gemm_i8_row_stride_bytes(Dim),
        },
        {
            KTSpm,
            Dim,
            SeqK,
            npu_mpu_gemm_i8_row_stride_bytes(SeqK),
        },
        {
            ScoresSpm,
            SeqQ,
            SeqK,
            npu_mpu_gemm_i32_row_stride_bytes(SeqK),
        },
        {
            PSpm,
            SeqQ,
            SeqK,
            npu_mpu_gemm_i8_row_stride_bytes(SeqK),
        },
        {
            VSpm,
            SeqK,
            DimV,
            npu_mpu_gemm_i8_row_stride_bytes(DimV),
        },
        {
            OutSpm,
            SeqQ,
            DimV,
            npu_mpu_gemm_i32_row_stride_bytes(DimV),
        },
        {
            QkTileM,
            QkTileN,
            QkTileK,
        },
        {
            PvTileM,
            PvTileN,
            PvTileK,
        },
        {
            DeviceId,
            SyncIndicator,
            SetCompletionSync,
            0U,
            0U,
            0U,
        },
        {
            DeviceId,
            SyncIndicator,
            SetCompletionSync,
            0U,
            0U,
            0U,
        },
        WaitTimeout,
    };

    std::array<int8_t, SeqQ * Dim> q_i8 = {};
    std::array<int8_t, SeqK * Dim> k_i8 = {};
    std::array<int8_t, Dim * SeqK> k_t_i8 = {};
    std::array<int8_t, SeqK * DimV> v_i8 = {};
    std::array<int8_t, SeqQ * SeqK> p_i8 = {};
    std::array<int32_t, SeqQ * SeqK> scores_i32 = {};
    std::array<int32_t, SeqQ * DimV> out_i32 = {};
    std::array<float, SeqQ * SeqK> scores_f32 = {};
    std::array<float, SeqQ> m_state = {};
    std::array<float, SeqQ> l_state = {};
    std::array<float, SeqQ * SeqK> p_f32 = {};
    std::array<float, SeqQ * DimV> out_f32 = {};
    std::array<PrimitiveScaleMetadata, SeqQ> q_metadata = {};
    std::array<PrimitiveScaleMetadata, SeqK> k_metadata = {};
    std::array<PrimitiveScaleMetadata, SeqQ> p_metadata = {};
    std::array<PrimitiveScaleMetadata, DimV> v_metadata = {};

    std::array<int8_t, SeqQ * Dim> expected_q_i8 = {};
    std::array<int8_t, SeqK * Dim> expected_k_i8 = {};
    std::array<int8_t, Dim * SeqK> expected_k_t_i8 = {};
    std::array<int8_t, SeqK * DimV> expected_v_i8 = {};
    std::array<int8_t, SeqQ * SeqK> expected_p_i8 = {};
    std::array<int32_t, SeqQ * SeqK> expected_scores_i32 = {};
    std::array<int32_t, SeqQ * DimV> expected_out_i32 = {};
    std::array<float, SeqQ * SeqK> expected_scores_f32 = {};
    std::array<float, SeqQ> expected_m_state = {};
    std::array<float, SeqQ> expected_l_state = {};
    std::array<float, SeqQ * SeqK> expected_p_f32 = {};
    std::array<float, SeqQ * DimV> expected_out_f32 = {};
    std::array<PrimitiveScaleMetadata, SeqQ> expected_q_metadata = {};
    std::array<PrimitiveScaleMetadata, SeqK> expected_k_metadata = {};
    std::array<PrimitiveScaleMetadata, SeqQ> expected_p_metadata = {};
    std::array<PrimitiveScaleMetadata, DimV> expected_v_metadata = {};

    std::array<uint32_t, SeqQ * Dim> q_bits = {};
    std::array<uint32_t, SeqK * Dim> k_bits = {};
    std::array<uint32_t, SeqK * DimV> v_bits = {};
    std::array<uint32_t, SeqQ * SeqK> golden_scores_bits = {};
    std::array<uint32_t, SeqQ * SeqK> golden_prob_bits = {};
    std::array<uint32_t, SeqQ * DimV> golden_out_bits = {};
    std::array<float, SeqQ * SeqK> golden_scores_f32 = {};
    std::array<float, SeqQ * SeqK> golden_prob_f32 = {};
    std::array<float, SeqQ * DimV> golden_out_f32 = {};
    std::array<uint32_t, SeqQ * SeqK> expected_scores_bits = {};
    std::array<uint32_t, SeqQ * SeqK> expected_p_bits = {};
    std::array<uint32_t, SeqQ * DimV> expected_out_bits = {};
    std::array<uint32_t, SeqQ> expected_m_bits = {};
    std::array<uint32_t, SeqQ> expected_l_bits = {};
    std::array<uint32_t, SeqQ * SeqK> scores_f32_bits = {};
    std::array<uint32_t, SeqQ> m_state_bits = {};
    std::array<uint32_t, SeqQ> l_state_bits = {};
    std::array<uint32_t, SeqQ * SeqK> p_f32_bits = {};
    std::array<uint32_t, SeqQ * DimV> out_f32_bits = {};

    const NpuFlashAttentionQkShape qk_shape =
        npu_flash_attention_qk_shape_from_basic(&shape);
    const NpuFlashAttentionPvShape pv_shape =
        npu_flash_attention_pv_shape_from_basic(&shape);
    const float scaling = npu_golden_flashattention_default_scale_f32(Dim);

    if (strcmp(scenario, "flash_attention_basic_f32") != 0) {
        printf("FLASH_ATTENTION_BASIC_UNKNOWN_SCENARIO=%s\n", scenario);
        return 1;
    }

    packFloatBits(q_f32.data(), q_f32.size(), q_bits.data());
    packFloatBits(k_f32.data(), k_f32.size(), k_bits.data());
    packFloatBits(v_f32.data(), v_f32.size(), v_bits.data());
    npu_golden_flashattention_f32(
        q_bits.data(), k_bits.data(), v_bits.data(), SeqQ, SeqK, Dim, DimV,
        scaling, golden_scores_bits.data(), golden_prob_bits.data(),
        golden_out_bits.data());
    unpackFloatBits(golden_scores_bits.data(), golden_scores_bits.size(),
                    golden_scores_f32.data());
    unpackFloatBits(golden_prob_bits.data(), golden_prob_bits.size(),
                    golden_prob_f32.data());
    unpackFloatBits(golden_out_bits.data(), golden_out_bits.size(),
                    golden_out_f32.data());

    npu_flash_attention_qk_prepare_operands(
        q_f32.data(), k_f32.data(), &qk_shape, expected_q_i8.data(),
        expected_q_metadata.data(), expected_k_i8.data(),
        expected_k_metadata.data(), expected_k_t_i8.data());
    npu_golden_mpu_gemm_i8_i8_i32(
        expected_q_i8.data(), expected_k_t_i8.data(),
        expected_scores_i32.data(), SeqQ, SeqK, Dim, Dim, SeqK, SeqK);
    npu_flash_attention_qk_dequantize_scores_f32(
        expected_scores_i32.data(), expected_q_metadata.data(),
        expected_k_metadata.data(), &qk_shape, expected_scores_f32.data());
    npu_flash_attention_init_online_softmax_state(
        &shape, expected_m_state.data(), expected_l_state.data());
    primitiveOnlineSoftmaxF32(
        expected_scores_f32.data(), SeqQ, SeqK, expected_m_state.data(),
        expected_l_state.data(), expected_m_state.data(),
        expected_l_state.data(), expected_p_f32.data());
    npu_flash_attention_pv_prepare_operands(
        expected_p_f32.data(), v_f32.data(), &pv_shape, expected_p_i8.data(),
        expected_p_metadata.data(), expected_v_i8.data(),
        expected_v_metadata.data());
    npu_golden_mpu_gemm_i8_i8_i32(
        expected_p_i8.data(), expected_v_i8.data(), expected_out_i32.data(),
        SeqQ, DimV, SeqK, SeqK, DimV, DimV);
    npu_flash_attention_pv_dequantize_outputs_f32(
        expected_out_i32.data(), expected_p_metadata.data(),
        expected_v_metadata.data(), &pv_shape, expected_out_f32.data());
    packFloatBits(expected_scores_f32.data(), expected_scores_f32.size(),
                  expected_scores_bits.data());
    packFloatBits(expected_p_f32.data(), expected_p_f32.size(),
                  expected_p_bits.data());
    packFloatBits(expected_out_f32.data(), expected_out_f32.size(),
                  expected_out_bits.data());
    packFloatBits(expected_m_state.data(), expected_m_state.size(),
                  expected_m_bits.data());
    packFloatBits(expected_l_state.data(), expected_l_state.size(),
                  expected_l_bits.data());

    if (npu_expect_float_vector_close(
            "FLASH_ATTENTION_BASIC_REFERENCE_SCORES_F32",
            golden_scores_bits.data(), expected_scores_bits.data(),
            SeqQ * SeqK, AbsTolerance, RelTolerance) != 0 ||
        npu_expect_float_vector_close(
            "FLASH_ATTENTION_BASIC_REFERENCE_PROBS_F32",
            golden_prob_bits.data(), expected_p_bits.data(),
            SeqQ * SeqK, AbsTolerance, RelTolerance) != 0 ||
        npu_expect_float_vector_close(
            "FLASH_ATTENTION_BASIC_REFERENCE_OUT_F32",
            golden_out_bits.data(), expected_out_bits.data(),
            SeqQ * DimV, AbsTolerance, RelTolerance) != 0) {
        return 1;
    }

    const NpuFlashAttentionBuffers buffers = {
        q_f32.data(),
        k_f32.data(),
        v_f32.data(),
        expected_scores_i32.data(),
        expected_out_i32.data(),
        q_i8.data(),
        k_i8.data(),
        k_t_i8.data(),
        v_i8.data(),
        scores_i32.data(),
        scores_f32.data(),
        m_state.data(),
        l_state.data(),
        p_f32.data(),
        p_i8.data(),
        out_i32.data(),
        out_f32.data(),
        q_metadata.data(),
        k_metadata.data(),
        p_metadata.data(),
        v_metadata.data(),
    };

    const int status =
        npu_flash_attention_run_basic_f32(&shape, &buffers, &exec);
    if (expectStatus("FLASH_ATTENTION_BASIC_STATUS", status,
                     NPU_FLASH_ATTENTION_OK) != 0) {
        return 1;
    }

    packFloatBits(scores_f32.data(), scores_f32.size(),
                  scores_f32_bits.data());
    packFloatBits(m_state.data(), m_state.size(), m_state_bits.data());
    packFloatBits(l_state.data(), l_state.size(), l_state_bits.data());
    packFloatBits(p_f32.data(), p_f32.size(), p_f32_bits.data());
    packFloatBits(out_f32.data(), out_f32.size(), out_f32_bits.data());

    printI32Vector("FLASH_ATTENTION_BASIC_SCORES_I32", scores_i32.data(),
                   scores_i32.size());
    printF32Vector("FLASH_ATTENTION_BASIC_SCORES_F32", scores_f32.data(),
                   scores_f32.size());
    printF32Vector("FLASH_ATTENTION_BASIC_M_STATE", m_state.data(),
                   m_state.size());
    printF32Vector("FLASH_ATTENTION_BASIC_L_STATE", l_state.data(),
                   l_state.size());
    printF32Vector("FLASH_ATTENTION_BASIC_PROBS_F32", p_f32.data(),
                   p_f32.size());
    printI32Vector("FLASH_ATTENTION_BASIC_OUT_I32", out_i32.data(),
                   out_i32.size());
    printF32Vector("FLASH_ATTENTION_BASIC_OUT_F32", out_f32.data(),
                   out_f32.size());
    printF32Vector("FLASH_ATTENTION_BASIC_GOLDEN_SCORES_F32",
                   golden_scores_f32.data(), golden_scores_f32.size());
    printF32Vector("FLASH_ATTENTION_BASIC_GOLDEN_PROBS_F32",
                   golden_prob_f32.data(), golden_prob_f32.size());
    printF32Vector("FLASH_ATTENTION_BASIC_GOLDEN_OUT_F32",
                   golden_out_f32.data(), golden_out_f32.size());

    if (npu_expect_u32_vector(
            "FLASH_ATTENTION_BASIC_SCORES_I32",
            reinterpret_cast<const uint32_t *>(expected_scores_i32.data()),
            reinterpret_cast<const uint32_t *>(scores_i32.data()),
            SeqQ * SeqK) != 0 ||
        npu_expect_float_vector_close(
            "FLASH_ATTENTION_BASIC_SCORES_F32", golden_scores_bits.data(),
            scores_f32_bits.data(), SeqQ * SeqK, AbsTolerance,
            RelTolerance) != 0 ||
        npu_expect_float_vector_close(
            "FLASH_ATTENTION_BASIC_M_STATE", expected_m_bits.data(),
            m_state_bits.data(), SeqQ, AbsTolerance, RelTolerance) != 0 ||
        npu_expect_float_vector_close(
            "FLASH_ATTENTION_BASIC_L_STATE", expected_l_bits.data(),
            l_state_bits.data(), SeqQ, AbsTolerance, RelTolerance) != 0 ||
        npu_expect_float_vector_close(
            "FLASH_ATTENTION_BASIC_PROBS_F32", golden_prob_bits.data(),
            p_f32_bits.data(), SeqQ * SeqK, AbsTolerance,
            RelTolerance) != 0 ||
        npu_expect_u32_vector(
            "FLASH_ATTENTION_BASIC_OUT_I32",
            reinterpret_cast<const uint32_t *>(expected_out_i32.data()),
            reinterpret_cast<const uint32_t *>(out_i32.data()),
            SeqQ * DimV) != 0 ||
        npu_expect_float_vector_close(
            "FLASH_ATTENTION_BASIC_OUT_F32", golden_out_bits.data(),
            out_f32_bits.data(), SeqQ * DimV, AbsTolerance,
            RelTolerance) != 0) {
        return 1;
    }

    printf("FLASH_ATTENTION_BASIC_F32_PASS\n");
    return 0;
}
