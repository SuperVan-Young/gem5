/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <array>

#include "flash_attention.hh"
#include "golden/mpu_gemm.hh"
#include "mpu_gemm.hh"
#include "npu_assert.hh"

namespace
{

static constexpr uint32_t SeqQ = 3U;
static constexpr uint32_t SeqK = 5U;
static constexpr uint32_t DimV = 4U;
static constexpr uint32_t TileM = 2U;
static constexpr uint32_t TileN = 2U;
static constexpr uint32_t TileK = SeqK;
static constexpr uintptr_t PSpm = 0x6000b000ULL;
static constexpr uintptr_t VSpm = 0x6000c000ULL;
static constexpr uintptr_t OutSpm = 0x6000d000ULL;
static constexpr uint32_t DeviceId = 0U;
static constexpr uint32_t SyncIndicator = 0U;
static constexpr uint32_t SetCompletionSync = 0U;
static constexpr uint64_t WaitTimeout = 80000000ULL;
static constexpr float AbsTolerance = 1.0e-4f;
static constexpr float RelTolerance = 1.0e-5f;

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

static void
computeGoldenPvOutput(const float *p_f32, const float *v_f32, float *o_f32)
{
    for (uint32_t row = 0U; row < SeqQ; ++row) {
        for (uint32_t col = 0U; col < DimV; ++col) {
            float acc = 0.0f;
            for (uint32_t key = 0U; key < SeqK; ++key) {
                acc += p_f32[row * SeqK + key] * v_f32[key * DimV + col];
            }
            o_f32[row * DimV + col] = acc;
        }
    }
}

} // namespace

int
main(int argc, char **argv)
{
    const char *scenario = argc > 1 ? argv[1] : "flash_attention_pv_path";
    const std::array<float, SeqQ * SeqK> p_f32 = {
        127.0f / 254.0f, 64.0f / 254.0f, 32.0f / 254.0f, 16.0f / 254.0f,
        15.0f / 254.0f, 127.0f / 254.0f, 50.0f / 254.0f, 40.0f / 254.0f,
        20.0f / 254.0f, 17.0f / 254.0f, 127.0f / 254.0f, 70.0f / 254.0f,
        30.0f / 254.0f, 20.0f / 254.0f, 7.0f / 254.0f,
    };
    const std::array<float, SeqK * DimV> v_f32 = {
        127.0f, -64.0f, 32.0f, 0.0f,
        64.0f, 127.0f, -32.0f, 16.0f,
        -127.0f, 32.0f, 64.0f, -16.0f,
        32.0f, -127.0f, 127.0f, 48.0f,
        0.0f, 64.0f, -64.0f, -127.0f,
    };
    std::array<int8_t, SeqQ * SeqK> p_i8 = {};
    std::array<int8_t, SeqK * DimV> v_i8 = {};
    std::array<PrimitiveScaleMetadata, SeqQ> p_metadata = {};
    std::array<PrimitiveScaleMetadata, DimV> v_metadata = {};
    std::array<int32_t, SeqQ * DimV> expected_out_i32 = {};
    std::array<int32_t, SeqQ * DimV> actual_out_i32 = {};
    std::array<float, SeqQ * DimV> out_f32 = {};
    std::array<float, SeqQ * DimV> golden_o_f32 = {};
    std::array<uint32_t, SeqQ * DimV> out_f32_bits = {};
    std::array<uint32_t, SeqQ * DimV> golden_o_bits = {};
    const NpuFlashAttentionPvShape pv_shape = {
        SeqQ,
        SeqK,
        DimV,
    };
    const NpuMpuGemmProblemShape problem =
        npu_flash_attention_pv_gemm_problem(&pv_shape);
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
    const NpuMpuGemmSpmI8Matrix p_matrix = {
        PSpm,
        SeqQ,
        SeqK,
        npu_mpu_gemm_i8_row_stride_bytes(SeqK),
    };
    const NpuMpuGemmSpmI8Matrix v_matrix = {
        VSpm,
        SeqK,
        DimV,
        npu_mpu_gemm_i8_row_stride_bytes(DimV),
    };
    const NpuMpuGemmSpmI32Matrix out_matrix = {
        OutSpm,
        SeqQ,
        DimV,
        npu_mpu_gemm_i32_row_stride_bytes(DimV),
    };

    if (strcmp(scenario, "flash_attention_pv_path") != 0) {
        printf("FLASH_ATTENTION_PV_UNKNOWN_SCENARIO=%s\n", scenario);
        return 1;
    }

    if (expectStatus("FLASH_ATTENTION_PV_GEMM_M", (int)problem.m,
                     (int)SeqQ) != 0 ||
        expectStatus("FLASH_ATTENTION_PV_GEMM_N", (int)problem.n,
                     (int)DimV) != 0 ||
        expectStatus("FLASH_ATTENTION_PV_GEMM_K", (int)problem.k,
                     (int)SeqK) != 0) {
        return 1;
    }

    npu_flash_attention_pv_prepare_operands(
        p_f32.data(), v_f32.data(), &pv_shape, p_i8.data(), p_metadata.data(),
        v_i8.data(), v_metadata.data());

    npu_golden_mpu_gemm_i8_i8_i32(p_i8.data(), v_i8.data(),
                                  expected_out_i32.data(), SeqQ, DimV, SeqK,
                                  SeqK, DimV, DimV);
    computeGoldenPvOutput(p_f32.data(), v_f32.data(), golden_o_f32.data());

    npu_mpu_gemm_store_i8_matrix_to_spm(p_i8.data(), &p_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(v_i8.data(), &v_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&out_matrix, -1);

    if (npu_mpu_gemm_launch_tiled_mn(&p_matrix, &v_matrix, &out_matrix,
                                     &problem, &tile, &launch) != 0) {
        printf("FLASH_ATTENTION_PV_GEMM_LAUNCH_FAIL\n");
        return 1;
    }

    if (npu_wait_u32_vector_match(
            NULL, (volatile uint32_t *)(uintptr_t)out_matrix.base_addr,
            (const uint32_t *)expected_out_i32.data(), SeqQ * DimV,
            WaitTimeout) != 0) {
        npu_mpu_gemm_load_i32_matrix_from_spm(&out_matrix,
                                              actual_out_i32.data());
        printf("FLASH_ATTENTION_PV_OUT_I32_FAIL\n");
        npu_expect_u32_vector("FLASH_ATTENTION_PV_OUT_I32",
                              (const uint32_t *)expected_out_i32.data(),
                              (const uint32_t *)actual_out_i32.data(),
                              SeqQ * DimV);
        return 1;
    }
    npu_mpu_gemm_load_i32_matrix_from_spm(&out_matrix, actual_out_i32.data());

    npu_flash_attention_pv_dequantize_outputs_f32(
        actual_out_i32.data(), p_metadata.data(), v_metadata.data(),
        &pv_shape, out_f32.data());
    packFloatBits(out_f32.data(), out_f32.size(), out_f32_bits.data());
    packFloatBits(golden_o_f32.data(), golden_o_f32.size(),
                  golden_o_bits.data());

    printI32Vector("FLASH_ATTENTION_PV_OUT_I32", actual_out_i32.data(),
                   actual_out_i32.size());
    printF32Vector("FLASH_ATTENTION_PV_OUT_F32", out_f32.data(),
                   out_f32.size());
    printF32Vector("FLASH_ATTENTION_PV_GOLDEN_O_F32", golden_o_f32.data(),
                   golden_o_f32.size());

    if (npu_expect_u32_vector("FLASH_ATTENTION_PV_OUT_I32",
                              (const uint32_t *)expected_out_i32.data(),
                              (const uint32_t *)actual_out_i32.data(),
                              SeqQ * DimV) != 0) {
        return 1;
    }
    if (npu_expect_float_vector_close("FLASH_ATTENTION_PV_OUT_F32",
                                      golden_o_bits.data(),
                                      out_f32_bits.data(), SeqQ * DimV,
                                      AbsTolerance, RelTolerance) != 0) {
        return 1;
    }

    printf("FLASH_ATTENTION_PV_PATH_PASS\n");
    return 0;
}
