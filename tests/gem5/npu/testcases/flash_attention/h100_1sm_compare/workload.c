/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <gem5/m5ops.h>

#include "flash_attention.hh"
#include "mpu_gemm.hh"
#include "npu_assert.hh"

namespace
{

static constexpr uint32_t Batch = 1U;
static constexpr uint32_t Heads = 1U;
static constexpr uint32_t Sms = 1U;
static constexpr uint32_t SeqLen = 1024U;
static constexpr uint32_t HeadDim = 128U;
static constexpr uint32_t DimV = HeadDim;
static constexpr uint32_t TileDim = 128U;
static constexpr uint32_t ProbeM = 2U;
static constexpr uint32_t ProbeN = 3U;
static constexpr uint32_t ProbeK = 4U;
static constexpr uintptr_t QSpm = 0x60000000ULL;
static constexpr uintptr_t KTSpm = 0x60004000ULL;
static constexpr uintptr_t ScoresSpm = 0x60008000ULL;
static constexpr uintptr_t PSpm = 0x60018000ULL;
static constexpr uintptr_t VSpm = 0x6001c000ULL;
static constexpr uintptr_t OutSpm = 0x60020000ULL;
static constexpr uintptr_t ProbeASpm = 0x60030000ULL;
static constexpr uintptr_t ProbeBSpm = 0x60031000ULL;
static constexpr uintptr_t ProbeCSpm = 0x60032000ULL;
static constexpr uint32_t DeviceId = 0U;
static constexpr uint32_t ProbeSyncIndicator = 0x11U;
static constexpr uint32_t CompareQkSyncIndicator = 0x21U;
static constexpr uint32_t ComparePvSyncIndicator = 0x31U;
static constexpr uint32_t SetCompletionSync = 0U;
static constexpr uint64_t WaitTimeout = 80000000ULL;
static constexpr uint32_t SampleRows = 8U;

static constexpr const char *ExpectedScenario =
    "flash_attention_h100_1sm_compare";

static int
expectStatus(const char *label, int actual, int expected)
{
    if (actual == expected) {
        return 0;
    }

    printf("%s actual=%d expected=%d\n", label, actual, expected);
    return -1;
}

static uint64_t
gemmFlops(uint32_t m, uint32_t n, uint32_t k)
{
    return 2ULL * (uint64_t)m * (uint64_t)n * (uint64_t)k;
}

static void
referenceGemmI8I8I32(const int8_t *a_data, const int8_t *b_data,
                     int32_t *c_data, uint32_t m, uint32_t n, uint32_t k)
{
    for (uint32_t row = 0U; row < m; ++row) {
        for (uint32_t col = 0U; col < n; ++col) {
            int32_t acc = 0;
            for (uint32_t depth = 0U; depth < k; ++depth) {
                acc += (int32_t)a_data[row * k + depth] *
                       (int32_t)b_data[depth * n + col];
            }
            c_data[row * n + col] = acc;
        }
    }
}

static void
fillQkvInputs(float *q_f32, float *k_f32, float *v_f32)
{
    for (uint32_t row = 0U; row < SeqLen; ++row) {
        for (uint32_t col = 0U; col < HeadDim; ++col) {
            const int32_t q_code =
                (int32_t)((row * 17U + col * 13U) % 29U) - 14;
            const int32_t k_code =
                (int32_t)((row * 11U + col * 7U) % 31U) - 15;
            const int32_t v_code =
                (int32_t)((row * 5U + col * 19U) % 37U) - 18;
            q_f32[row * HeadDim + col] = (float)q_code / 14.0f;
            k_f32[row * HeadDim + col] = (float)k_code / 15.0f;
            v_f32[row * DimV + col] = (float)v_code / 18.0f;
        }
    }
}

static int
validateFiniteOutput(const float *out_f32, uint32_t rows, uint32_t cols)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        for (uint32_t col = 0U; col < cols; ++col) {
            const float value = out_f32[row * cols + col];
            if (!std::isfinite(value)) {
                printf(
                    "FLASH_ATTENTION_H100_1SM_COMPARE_OUTPUT_NONFINITE "
                    "row=%u col=%u value=%f\n",
                    row, col, (double)value);
                return -1;
            }
        }
    }
    return 0;
}

static float
sampleChecksum(const float *values, uint32_t rows, uint32_t cols)
{
    const uint32_t limit = rows < SampleRows ? rows : SampleRows;
    float checksum = 0.0f;

    for (uint32_t row = 0U; row < limit; ++row) {
        checksum += values[row * cols + (row % cols)];
    }
    return checksum;
}

} // namespace

int
main(int argc, char **argv)
{
    const char *scenario = argc > 1 ? argv[1] : ExpectedScenario;
    const NpuFlashAttentionShape shape = {
        SeqLen,
        SeqLen,
        HeadDim,
        DimV,
    };
    const NpuFlashAttentionQkShape qk_shape =
        npu_flash_attention_qk_shape_from_basic(&shape);
    const NpuFlashAttentionPvShape pv_shape =
        npu_flash_attention_pv_shape_from_basic(&shape);
    const NpuMpuGemmProblemShape qk_problem =
        npu_flash_attention_qk_gemm_problem(&qk_shape);
    const NpuMpuGemmProblemShape pv_problem =
        npu_flash_attention_pv_gemm_problem(&pv_shape);
    const float qk_scale = npu_flash_attention_qk_scale_f32(shape.dim);
    const uint64_t qk_flops =
        gemmFlops(qk_problem.m, qk_problem.n, qk_problem.k);
    const uint64_t pv_flops =
        gemmFlops(pv_problem.m, pv_problem.n, pv_problem.k);
    const uint64_t total_flops = qk_flops + pv_flops;
    const std::array<int8_t, ProbeM * ProbeK> probe_a = {
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    const std::array<int8_t, ProbeK * ProbeN> probe_b = {
        1, 0, 2,
        -1, 3, 1,
        2, 1, 0,
        1, -2, 4,
    };
    std::array<int32_t, ProbeM * ProbeN> probe_expected = {};
    std::array<int32_t, ProbeM * ProbeN> probe_actual = {};
    const NpuMpuGemmSpmI8Matrix probe_a_matrix = {
        ProbeASpm,
        ProbeM,
        ProbeK,
        npu_mpu_gemm_i8_row_stride_bytes(ProbeK),
    };
    const NpuMpuGemmSpmI8Matrix probe_b_matrix = {
        ProbeBSpm,
        ProbeK,
        ProbeN,
        npu_mpu_gemm_i8_row_stride_bytes(ProbeN),
    };
    const NpuMpuGemmSpmI32Matrix probe_c_matrix = {
        ProbeCSpm,
        ProbeM,
        ProbeN,
        npu_mpu_gemm_i32_row_stride_bytes(ProbeN),
    };
    const NpuMpuGemmProblemShape probe_problem = {
        ProbeM,
        ProbeN,
        ProbeK,
    };
    const NpuMpuGemmTileShape probe_tile = {
        ProbeM,
        ProbeN,
        ProbeK,
    };
    const NpuMpuGemmLaunchConfig probe_launch = {
        DeviceId,
        ProbeSyncIndicator,
        SetCompletionSync,
        0U,
        0U,
        0U,
    };
    const NpuFlashAttentionExecutionConfig exec = {
        {
            QSpm,
            TileDim,
            HeadDim,
            npu_mpu_gemm_i8_row_stride_bytes(HeadDim),
        },
        {
            KTSpm,
            HeadDim,
            TileDim,
            npu_mpu_gemm_i8_row_stride_bytes(TileDim),
        },
        {
            ScoresSpm,
            TileDim,
            TileDim,
            npu_mpu_gemm_i32_row_stride_bytes(TileDim),
        },
        {
            PSpm,
            TileDim,
            TileDim,
            npu_mpu_gemm_i8_row_stride_bytes(TileDim),
        },
        {
            VSpm,
            TileDim,
            TileDim,
            npu_mpu_gemm_i8_row_stride_bytes(TileDim),
        },
        {
            OutSpm,
            TileDim,
            TileDim,
            npu_mpu_gemm_i32_row_stride_bytes(TileDim),
        },
        {
            TileDim,
            TileDim,
            HeadDim,
        },
        {
            TileDim,
            TileDim,
            TileDim,
        },
        {
            DeviceId,
            CompareQkSyncIndicator,
            SetCompletionSync,
            0U,
            0U,
            0U,
        },
        {
            DeviceId,
            ComparePvSyncIndicator,
            SetCompletionSync,
            0U,
            0U,
            0U,
        },
        WaitTimeout,
    };
    std::vector<float> q_input(shape.seq_q * shape.dim);
    std::vector<float> k_input(shape.seq_k * shape.dim);
    std::vector<float> v_input(shape.seq_k * shape.dim_v);
    std::vector<int32_t> expected_scores_i32(shape.seq_q * shape.seq_k, 0);
    std::vector<int32_t> expected_out_i32(shape.seq_q * shape.dim_v, 0);
    std::vector<int8_t> q_i8(shape.seq_q * shape.dim, 0);
    std::vector<int8_t> k_i8(shape.seq_k * shape.dim, 0);
    std::vector<int8_t> k_t_i8(shape.dim * shape.seq_k, 0);
    std::vector<int8_t> v_i8(shape.seq_k * shape.dim_v, 0);
    std::vector<int32_t> scores_i32(shape.seq_q * shape.seq_k, 0);
    std::vector<float> scores_f32(shape.seq_q * shape.seq_k, 0.0f);
    std::vector<float> m_state(shape.seq_q, 0.0f);
    std::vector<float> l_state(shape.seq_q, 0.0f);
    std::vector<float> p_f32(shape.seq_q * shape.seq_k, 0.0f);
    std::vector<int8_t> p_i8(shape.seq_q * shape.seq_k, 0);
    std::vector<int32_t> out_i32(shape.seq_q * shape.dim_v, 0);
    std::vector<float> out_f32(shape.seq_q * shape.dim_v, 0.0f);
    std::vector<PrimitiveScaleMetadata> q_metadata(shape.seq_q);
    std::vector<PrimitiveScaleMetadata> k_metadata(shape.seq_k);
    std::vector<PrimitiveScaleMetadata> p_metadata(shape.seq_q);
    std::vector<PrimitiveScaleMetadata> v_metadata(shape.dim_v);
    std::vector<int8_t> k_t_tile_i8(shape.dim * TileDim, 0);
    std::vector<int8_t> p_tile_i8(TileDim * TileDim, 0);
    std::vector<int8_t> v_tile_i8(TileDim * TileDim, 0);
    std::vector<int32_t> tile_i32(TileDim * TileDim, 0);
    const NpuFlashAttentionBuffers buffers = {
        q_input.data(),
        k_input.data(),
        v_input.data(),
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
        k_t_tile_i8.data(),
        p_tile_i8.data(),
        v_tile_i8.data(),
        tile_i32.data(),
    };

    if (strcmp(scenario, ExpectedScenario) != 0) {
        printf("FLASH_ATTENTION_H100_1SM_COMPARE_UNKNOWN_SCENARIO=%s\n",
               scenario);
        return 1;
    }

    if (expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_BATCH", (int)Batch,
                     1) != 0 ||
        expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_HEADS", (int)Heads,
                     1) != 0 ||
        expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_SMS", (int)Sms, 1) !=
            0 ||
        expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_QK_M",
                     (int)qk_problem.m, (int)SeqLen) != 0 ||
        expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_QK_N",
                     (int)qk_problem.n, (int)SeqLen) != 0 ||
        expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_QK_K",
                     (int)qk_problem.k, (int)HeadDim) != 0 ||
        expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_PV_M",
                     (int)pv_problem.m, (int)SeqLen) != 0 ||
        expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_PV_N",
                     (int)pv_problem.n, (int)DimV) != 0 ||
        expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_PV_K",
                     (int)pv_problem.k, (int)SeqLen) != 0) {
        return 1;
    }

    npu_mpu_gemm_store_i8_matrix_to_spm(probe_a.data(), &probe_a_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(probe_b.data(), &probe_b_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&probe_c_matrix, -1);
    referenceGemmI8I8I32(probe_a.data(), probe_b.data(),
                         probe_expected.data(), ProbeM, ProbeN, ProbeK);

    if (npu_mpu_gemm_launch_single_tile(&probe_a_matrix, &probe_b_matrix,
                                        &probe_c_matrix, &probe_problem,
                                        &probe_tile, &probe_launch, 0U,
                                        0U) != 0) {
        printf("FLASH_ATTENTION_H100_1SM_COMPARE_PROBE_LAUNCH_FAIL\n");
        return 1;
    }

    if (npu_wait_u32_vector_match(
            NULL,
            (volatile uint32_t *)(uintptr_t)probe_c_matrix.base_addr,
            (const uint32_t *)probe_expected.data(), ProbeM * ProbeN,
            WaitTimeout) != 0) {
        npu_mpu_gemm_load_i32_matrix_from_spm(&probe_c_matrix,
                                              probe_actual.data());
        printf("FLASH_ATTENTION_H100_1SM_COMPARE_PROBE_FAIL\n");
        npu_expect_u32_vector(
            "FLASH_ATTENTION_H100_1SM_COMPARE_PROBE",
            (const uint32_t *)probe_expected.data(),
            (const uint32_t *)probe_actual.data(), ProbeM * ProbeN);
        return 1;
    }

    printf("FLASH_ATTENTION_H100_1SM_COMPARE_SCENARIO=%s\n", scenario);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_SHAPE seq_len=%u head_dim=%u "
        "batch=%u heads=%u\n",
        shape.seq_q, shape.dim, Batch, Heads);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_CONTRACT sms=%u "
        "mapping=one_sm_per_head_per_batch\n",
        Sms);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_QK_GEMM m=%u n=%u k=%u scale=%.8f\n",
        qk_problem.m, qk_problem.n, qk_problem.k, (double)qk_scale);
    printf("FLASH_ATTENTION_H100_1SM_COMPARE_PV_GEMM m=%u n=%u k=%u\n",
           pv_problem.m, pv_problem.n, pv_problem.k);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_WORKLOAD "
        "scenario=%s seq_len=%u head_dim=%u qk_flops=%llu pv_flops=%llu "
        "total_flops=%llu pass=PASS\n",
        scenario, shape.seq_q, shape.dim, (unsigned long long)qk_flops,
        (unsigned long long)pv_flops, (unsigned long long)total_flops);
    printf("FLASH_ATTENTION_H100_1SM_COMPARE_PROBE m=%u n=%u k=%u macs=%u "
           "sync_indicator=%u status=PASS\n",
           ProbeM, ProbeN, ProbeK, ProbeM * ProbeN * ProbeK,
           ProbeSyncIndicator);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_PROFILE_TAGS "
        "probe_sync=%u qk_sync=%u pv_sync=%u\n",
        ProbeSyncIndicator, CompareQkSyncIndicator, ComparePvSyncIndicator);

    m5_reset_stats(0U, 0U);
    fillQkvInputs(q_input.data(), k_input.data(), v_input.data());

    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_BEGIN "
        "scenario=%s compare_scope=full_flash_attention "
        "softmax_scope=cpu_helper tensor_path=qk_pv_only "
        "excluded_from_compare_latency=true\n",
        scenario);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_FLOPS "
        "qk_flops=%llu pv_flops=%llu total_flops=%llu "
        "tensor_path_only=true cpu_softmax_included=false\n",
        (unsigned long long)qk_flops, (unsigned long long)pv_flops,
        (unsigned long long)total_flops);

    const int status =
        npu_flash_attention_run_basic_f32(&shape, &buffers, &exec);
    if (expectStatus("FLASH_ATTENTION_H100_1SM_COMPARE_STATUS", status,
                     NPU_FLASH_ATTENTION_OK) != 0) {
        return 1;
    }
    if (validateFiniteOutput(out_f32.data(), shape.seq_q, shape.dim_v) != 0) {
        return 1;
    }

    const float checksum = sampleChecksum(out_f32.data(), shape.seq_q,
                                          shape.dim_v);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_OUTPUT "
        "sample_checksum=%.6f sample_row0_col0=%.6f "
        "sample_last=%.6f\n",
        (double)checksum, (double)out_f32[0],
        (double)out_f32[out_f32.size() - 1U]);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_END "
        "scenario=%s compare_scope=full_flash_attention "
        "softmax_scope=cpu_helper status=PASS\n",
        scenario);
    printf(
        "FLASH_ATTENTION_H100_1SM_COMPARE_NOTE "
        "helper_path=flash_attention.hh "
        "tensor_model=mpu_tile_scale_approx "
        "vector_model=shared_riscv_cpu_only\n");
    printf("FLASH_ATTENTION_H100_1SM_COMPARE_PASS\n");
    return 0;
}
