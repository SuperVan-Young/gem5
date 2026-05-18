/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <array>

#include "flash_attention.hh"
#include "golden/llm_f32.hh"
#include "golden/mpu_gemm.hh"
#include "mpu_gemm.hh"
#include "npu_assert.hh"

namespace
{

static constexpr uint32_t SeqQ = 3U;
static constexpr uint32_t SeqK = 5U;
static constexpr uint32_t Dim = 4U;
static constexpr uint32_t TileM = 2U;
static constexpr uint32_t TileN = 3U;
static constexpr uint32_t TileK = Dim;
static constexpr uintptr_t QSpm = 0x60008000ULL;
static constexpr uintptr_t KTSpm = 0x60009000ULL;
static constexpr uintptr_t ScoresSpm = 0x6000a000ULL;
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
    const char *scenario = argc > 1 ? argv[1] : "flash_attention_qk_path";
    const std::array<float, SeqQ * Dim> q_f32 = {
        63.5f, -31.5f, 16.0f, -63.5f,
        127.0f, -64.0f, 32.0f, -16.0f,
        31.75f, 0.25f, -10.5f, -31.75f,
    };
    const std::array<float, SeqK * Dim> k_f32 = {
        63.5f, -32.0f, 16.5f, -63.5f,
        -31.75f, 7.5f, 0.25f, 31.75f,
        127.0f, -64.0f, 32.0f, -16.0f,
        -15.875f, 8.0f, -4.0f, 15.875f,
        64.0f, 127.0f, -32.0f, 0.0f,
    };
    std::array<int8_t, SeqQ * Dim> q_i8 = {};
    std::array<int8_t, SeqK * Dim> k_i8 = {};
    std::array<int8_t, Dim * SeqK> k_t_i8 = {};
    std::array<PrimitiveScaleMetadata, SeqQ> q_metadata = {};
    std::array<PrimitiveScaleMetadata, SeqK> k_metadata = {};
    std::array<int32_t, SeqQ * SeqK> expected_scores_i32 = {};
    std::array<int32_t, SeqQ * SeqK> actual_scores_i32 = {};
    std::array<float, SeqQ * SeqK> scores_f32 = {};
    std::array<float, SeqQ * SeqK> golden_scores_f32 = {};
    std::array<uint32_t, SeqQ * Dim> q_bits = {};
    std::array<uint32_t, SeqK * Dim> k_bits = {};
    std::array<uint32_t, SeqQ * SeqK> scores_f32_bits = {};
    std::array<uint32_t, SeqQ * SeqK> golden_scores_bits = {};
    const NpuFlashAttentionQkShape qk_shape = {
        SeqQ,
        SeqK,
        Dim,
    };
    const NpuMpuGemmProblemShape problem =
        npu_flash_attention_qk_gemm_problem(&qk_shape);
    const NpuMpuGemmTileShape tile = {
        TileM,
        TileN,
        TileK,
    };
    const NpuMpuGemmLaunchConfig launch = {
        DeviceId,
        SyncIndicator,
        SetCompletionSync,
        0U,
        0U,
        0U,
    };
    const NpuMpuGemmSpmI8Matrix q_matrix = {
        QSpm,
        SeqQ,
        Dim,
        npu_mpu_gemm_i8_row_stride_bytes(Dim),
    };
    const NpuMpuGemmSpmI8Matrix k_t_matrix = {
        KTSpm,
        Dim,
        SeqK,
        npu_mpu_gemm_i8_row_stride_bytes(SeqK),
    };
    const NpuMpuGemmSpmI32Matrix scores_matrix = {
        ScoresSpm,
        SeqQ,
        SeqK,
        npu_mpu_gemm_i32_row_stride_bytes(SeqK),
    };
    const float score_scale = npu_flash_attention_qk_scale_f32(Dim);

    if (strcmp(scenario, "flash_attention_qk_path") != 0) {
        printf("FLASH_ATTENTION_QK_UNKNOWN_SCENARIO=%s\n", scenario);
        return 1;
    }

    if (expectStatus("FLASH_ATTENTION_QK_GEMM_M", (int)problem.m,
                     (int)SeqQ) != 0 ||
        expectStatus("FLASH_ATTENTION_QK_GEMM_N", (int)problem.n,
                     (int)SeqK) != 0 ||
        expectStatus("FLASH_ATTENTION_QK_GEMM_K", (int)problem.k,
                     (int)Dim) != 0) {
        return 1;
    }

    npu_flash_attention_qk_prepare_operands(
        q_f32.data(), k_f32.data(), &qk_shape, q_i8.data(), q_metadata.data(),
        k_i8.data(), k_metadata.data(), k_t_i8.data());
    for (uint32_t row = 0U; row < SeqK; ++row) {
        for (uint32_t col = 0U; col < Dim; ++col) {
            if (k_t_i8[col * SeqK + row] != k_i8[row * Dim + col]) {
                printf("FLASH_ATTENTION_QK_TRANSPOSE_FAIL row=%u col=%u\n",
                       row, col);
                return 1;
            }
        }
    }

    npu_golden_mpu_gemm_i8_i8_i32(q_i8.data(), k_t_i8.data(),
                                  expected_scores_i32.data(), SeqQ, SeqK, Dim,
                                  Dim, SeqK, SeqK);

    npu_mpu_gemm_store_i8_matrix_to_spm(q_i8.data(), &q_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(k_t_i8.data(), &k_t_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&scores_matrix, -1);

    if (npu_mpu_gemm_launch_tiled_mn(&q_matrix, &k_t_matrix, &scores_matrix,
                                     &problem, &tile, &launch) != 0) {
        printf("FLASH_ATTENTION_QK_GEMM_LAUNCH_FAIL\n");
        return 1;
    }

    if (npu_wait_u32_vector_match(
            NULL, (volatile uint32_t *)(uintptr_t)scores_matrix.base_addr,
            (const uint32_t *)expected_scores_i32.data(), SeqQ * SeqK,
            WaitTimeout) != 0) {
        npu_mpu_gemm_load_i32_matrix_from_spm(&scores_matrix,
                                              actual_scores_i32.data());
        printf("FLASH_ATTENTION_QK_SCORES_I32_FAIL\n");
        npu_expect_u32_vector("FLASH_ATTENTION_QK_SCORES_I32",
                              (const uint32_t *)expected_scores_i32.data(),
                              (const uint32_t *)actual_scores_i32.data(),
                              SeqQ * SeqK);
        return 1;
    }
    npu_mpu_gemm_load_i32_matrix_from_spm(&scores_matrix,
                                          actual_scores_i32.data());

    npu_flash_attention_qk_dequantize_scores_f32(
        actual_scores_i32.data(), q_metadata.data(), k_metadata.data(),
        &qk_shape, scores_f32.data());
    packFloatBits(q_f32.data(), q_f32.size(), q_bits.data());
    packFloatBits(k_f32.data(), k_f32.size(), k_bits.data());
    npu_golden_qk_scores_f32(q_bits.data(), k_bits.data(), SeqQ, SeqK, Dim,
                             score_scale, golden_scores_bits.data());
    for (uint32_t index = 0U; index < golden_scores_f32.size(); ++index) {
        golden_scores_f32[index] =
            npu_bits_to_float(golden_scores_bits[index]);
    }
    packFloatBits(scores_f32.data(), scores_f32.size(),
                  scores_f32_bits.data());

    printI32Vector("FLASH_ATTENTION_QK_SCORES_I32", actual_scores_i32.data(),
                   actual_scores_i32.size());
    printF32Vector("FLASH_ATTENTION_QK_SCORES_F32", scores_f32.data(),
                   scores_f32.size());
    printF32Vector("FLASH_ATTENTION_QK_GOLDEN_SCORES_F32",
                   golden_scores_f32.data(), golden_scores_f32.size());

    if (npu_expect_u32_vector("FLASH_ATTENTION_QK_SCORES_I32",
                              (const uint32_t *)expected_scores_i32.data(),
                              (const uint32_t *)actual_scores_i32.data(),
                              SeqQ * SeqK) != 0) {
        return 1;
    }
    if (npu_expect_float_vector_close(
            "FLASH_ATTENTION_QK_SCORES_F32", golden_scores_bits.data(),
            scores_f32_bits.data(), SeqQ * SeqK, AbsTolerance,
            RelTolerance) != 0) {
        return 1;
    }

    printf("FLASH_ATTENTION_QK_PATH_PASS\n");
    return 0;
}
