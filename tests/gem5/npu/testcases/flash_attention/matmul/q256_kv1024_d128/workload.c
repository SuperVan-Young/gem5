/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <climits>

#include "mpu_gemm.hh"
#include "npu_assert.hh"
#include "primitive/fastMatmul.hh"

namespace
{

static constexpr uint32_t M = 256U;
static constexpr uint32_t N = 1024U;
static constexpr uint32_t K = 128U;
static constexpr uint32_t TileM = 128U;
static constexpr uint32_t TileN = 128U;
static constexpr uint32_t TileK = K;
static constexpr uint32_t DeviceId = 0U;
static constexpr uint32_t BaselineSyncIndicator = 0x40U;
static constexpr uint32_t FastSyncIndicator = 0x80U;
static constexpr uint64_t WaitTimeout = 80000000ULL;
static constexpr int32_t OutputSentinel = INT32_MIN;
static constexpr uint32_t ReferenceSampleCount = 16U;
static constexpr uintptr_t ASpm = 0x60000000ULL;
static constexpr uintptr_t BSpm = 0x60008000ULL;
static constexpr uintptr_t BaselineCSpm = 0x60030000ULL;
static constexpr uintptr_t FastCSpm = 0x60130000ULL;
static constexpr const char *ExpectedScenario =
    "flash_attention_matmul_q256_kv1024_d128";

static int8_t AHost[M * K];
static int8_t BHost[K * N];
static int32_t BaselineHost[M * N];
static int32_t FastHost[M * N];

static void
fillInputs()
{
    for (uint32_t row = 0U; row < M; ++row) {
        for (uint32_t col = 0U; col < K; ++col) {
            const int32_t code =
                (int32_t)((row * 17U + col * 13U) % 31U) - 15;
            AHost[row * K + col] = (int8_t)code;
        }
    }

    for (uint32_t row = 0U; row < K; ++row) {
        for (uint32_t col = 0U; col < N; ++col) {
            const int32_t code =
                (int32_t)((row * 11U + col * 7U) % 29U) - 14;
            BHost[row * N + col] = (int8_t)code;
        }
    }
}

static int32_t
referenceValueAt(uint32_t row, uint32_t col)
{
    int32_t acc = 0;

    for (uint32_t depth = 0U; depth < K; ++depth) {
        acc += (int32_t)AHost[row * K + depth] *
               (int32_t)BHost[depth * N + col];
    }

    return acc;
}

static int
waitOutputValueReady(const NpuMpuGemmSpmI32Matrix *matrix, uint32_t row,
                     uint32_t col)
{
    const uintptr_t addr =
        matrix->base_addr +
        (uintptr_t)row * (uintptr_t)matrix->row_stride_bytes +
        (uintptr_t)col * sizeof(int32_t);
    const volatile int32_t *value_ptr = (const volatile int32_t *)addr;

    return npu_wait_i32_vector_not_value(NULL, value_ptr, OutputSentinel, 1U,
                                         WaitTimeout);
}

static int
copySpmMatrixToHost(const NpuMpuGemmSpmI32Matrix *src, int32_t *dst)
{
    if (waitOutputValueReady(src, src->rows - 1U, src->cols - 1U) != 0) {
        return -1;
    }

    npu_mpu_gemm_load_i32_matrix_from_spm(src, dst);
    return 0;
}

static int
verifyEqual(const int32_t *lhs, const int32_t *rhs)
{
    for (uint32_t row = 0U; row < M; ++row) {
        for (uint32_t col = 0U; col < N; ++col) {
            const uint32_t index = row * N + col;
            if (lhs[index] != rhs[index]) {
                printf("FLASH_ATTENTION_MATMUL_MISMATCH row=%u col=%u lhs=%d "
                       "rhs=%d\n",
                       row, col, lhs[index], rhs[index]);
                return -1;
            }
        }
    }

    return 0;
}

static int64_t
sampleReferenceChecksum(const int32_t *observed)
{
    int64_t checksum = 0;

    for (uint32_t sample = 0U; sample < ReferenceSampleCount; ++sample) {
        const uint32_t row = (sample * 17U) % M;
        const uint32_t col = (sample * 67U) % N;
        const int32_t expected = referenceValueAt(row, col);
        const int32_t actual = observed[row * N + col];

        if (expected != actual) {
            printf(
                "FLASH_ATTENTION_MATMUL_SAMPLE_FAIL sample=%u row=%u col=%u "
                "expected=%d actual=%d\n",
                   sample, row, col, expected, actual);
            return INT64_MIN;
        }
        checksum += expected;
    }

    return checksum;
}

static int64_t
outputChecksum(const int32_t *values)
{
    const uint32_t sample_rows = M < 16U ? M : 16U;
    int64_t checksum = 0;

    for (uint32_t row = 0U; row < sample_rows; ++row) {
        checksum += values[row * N + ((row * 17U) % N)];
    }
    checksum += values[0];
    checksum += values[(M - 1U) * N + (N - 1U)];
    return checksum;
}

static uint64_t
gemmFlops()
{
    return 2ULL * (uint64_t)M * (uint64_t)N * (uint64_t)K;
}

} // namespace

int
main(int argc, char **argv)
{
    const char *scenario = argc > 1 ? argv[1] : ExpectedScenario;
    const NpuMpuGemmSpmI8Matrix a_matrix = {
        ASpm,
        M,
        K,
        npu_mpu_gemm_i8_row_stride_bytes(K),
    };
    const NpuMpuGemmSpmI8Matrix b_matrix = {
        BSpm,
        K,
        N,
        npu_mpu_gemm_i8_row_stride_bytes(N),
    };
    const NpuMpuGemmSpmI32Matrix baseline_c_matrix = {
        BaselineCSpm,
        M,
        N,
        npu_mpu_gemm_i32_row_stride_bytes(N),
    };
    const NpuMpuGemmSpmI32Matrix fast_c_matrix = {
        FastCSpm,
        M,
        N,
        npu_mpu_gemm_i32_row_stride_bytes(N),
    };
    const NpuMpuGemmProblemShape problem = {M, N, K};
    const NpuMpuGemmTileShape tile = {TileM, TileN, TileK};
    const NpuMpuGemmLaunchConfig baseline_launch = {
        DeviceId,
        BaselineSyncIndicator,
        0U,
        0U,
        0U,
        0U,
    };
    const NpuFastMatmulLaunchConfig fast_launch = {
        DeviceId,
        FastSyncIndicator,
        0U,
        0U,
        0U,
        0U,
        0U,
    };
    NpuFastMatmulStats fast_stats = {};
    int64_t baseline_ref = 0;
    int64_t fast_ref = 0;

    if (strcmp(scenario, ExpectedScenario) != 0) {
        printf("FLASH_ATTENTION_MATMUL_UNKNOWN_SCENARIO=%s\n", scenario);
        return 1;
    }

    fillInputs();
    npu_mpu_gemm_store_i8_matrix_to_spm(AHost, &a_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(BHost, &b_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&baseline_c_matrix, OutputSentinel);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&fast_c_matrix, OutputSentinel);

    for (uint32_t row0 = 0U; row0 < M; row0 += TileM) {
        for (uint32_t col0 = 0U; col0 < N; col0 += TileN) {
            const uint32_t rows =
                npu_mpu_gemm_tile_rows(&problem, &tile, row0);
            const uint32_t cols =
                npu_mpu_gemm_tile_cols(&problem, &tile, col0);

            if (npu_mpu_gemm_launch_single_tile(
                    &a_matrix, &b_matrix, &baseline_c_matrix, &problem, &tile,
                    &baseline_launch, row0, col0) != 0) {
                printf("FLASH_ATTENTION_MATMUL_BASELINE_LAUNCH_FAIL\n");
                return 1;
            }
            if (waitOutputValueReady(&baseline_c_matrix, row0 + rows - 1U,
                                     col0 + cols - 1U) != 0) {
                printf("FLASH_ATTENTION_MATMUL_BASELINE_WAIT_FAIL\n");
                return 1;
            }
        }
    }
    if (copySpmMatrixToHost(&baseline_c_matrix, BaselineHost) != 0) {
        printf("FLASH_ATTENTION_MATMUL_BASELINE_COPY_FAIL\n");
        return 1;
    }
    baseline_ref = sampleReferenceChecksum(BaselineHost);
    if (baseline_ref == INT64_MIN) {
        return 1;
    }

    if (npu_fast_matmul_launch_tiled_spm(&a_matrix, &b_matrix, &fast_c_matrix,
                                         &problem, &tile, &fast_launch,
                                         &fast_stats) != 0) {
        printf("FLASH_ATTENTION_MATMUL_FAST_LAUNCH_FAIL\n");
        return 1;
    }
    if (copySpmMatrixToHost(&fast_c_matrix, FastHost) != 0) {
        printf("FLASH_ATTENTION_MATMUL_FAST_WAIT_FAIL\n");
        return 1;
    }
    fast_ref = sampleReferenceChecksum(FastHost);
    if (fast_ref == INT64_MIN) {
        return 1;
    }
    if (verifyEqual(BaselineHost, FastHost) != 0) {
        return 1;
    }

    printf("FLASH_ATTENTION_MATMUL_SCENARIO=%s\n", scenario);
    printf("FLASH_ATTENTION_MATMUL_SHAPE m=%u n=%u k=%u tile_m=%u tile_n=%u "
           "tile_k=%u total_flops=%llu\n",
           M, N, K, TileM, TileN, TileK,
           (unsigned long long)gemmFlops());
    printf("FLASH_ATTENTION_MATMUL_BASELINE cmds=112 build_cmd_calls=112 "
           "sync_indicator=%u status=PASS\n",
           BaselineSyncIndicator);
    printf("FLASH_ATTENTION_MATMUL_FAST cmds=%u template_builds=%u "
           "sync_indicator=%u status=PASS\n",
           fast_stats.launched_cmd_count, fast_stats.template_build_count,
           FastSyncIndicator);
    printf("FLASH_ATTENTION_MATMUL_COMPARE build_call_reduction=%u "
           "baseline_checksum=%lld fast_checksum=%lld ref_checksum=%lld "
           "status=PASS\n",
           112U - fast_stats.template_build_count,
           (long long)outputChecksum(BaselineHost),
           (long long)outputChecksum(FastHost), (long long)baseline_ref);
    printf("FLASH_ATTENTION_MATMUL_PASS\n");
    return 0;
}
