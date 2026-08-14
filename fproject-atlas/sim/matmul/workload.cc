/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <climits>
#include <vector>

#include "mpu_gemm_layout.hh"
#include "npu_assert.hh"
#include "primitive/fastMatmul.hh"

namespace
{

constexpr uint64_t WaitTimeout = 80000000ULL;
constexpr int32_t OutputSentinel = INT32_MIN;

uint32_t
parseU32(const char *text, const char *name)
{
    char *end = nullptr;
    const unsigned long value = strtoul(text, &end, 0);
    if (text[0] == '\0' || *end != '\0' || value == 0 || value > UINT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return static_cast<uint32_t>(value);
}

uintptr_t
align64(uintptr_t value)
{
    return (value + 63U) & ~static_cast<uintptr_t>(63U);
}

uint64_t
paddedElements(uint32_t rows, uint32_t cols, uint32_t block_rows,
               uint32_t block_cols)
{
    const uint64_t row_blocks = (rows + block_rows - 1U) / block_rows;
    const uint64_t col_blocks = (cols + block_cols - 1U) / block_cols;
    return row_blocks * col_blocks * block_rows * block_cols;
}

int8_t
aValue(uint32_t row, uint32_t col)
{
    return static_cast<int8_t>((row * 17U + col * 13U) % 31U - 15U);
}

int8_t
bValue(uint32_t row, uint32_t col)
{
    return static_cast<int8_t>((row * 11U + col * 7U) % 29U - 14U);
}

int32_t
referenceValue(uint32_t row, uint32_t col, uint32_t k)
{
    int32_t result = 0;
    for (uint32_t depth = 0; depth < k; ++depth) {
        result += static_cast<int32_t>(aValue(row, depth)) *
                  static_cast<int32_t>(bValue(depth, col));
    }
    return result;
}

} // namespace

int
main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "usage: %s M N K ARRAY_DIM SPM_BASE SPM_SIZE\n", argv[0]);
        return 2;
    }
    const uint32_t m = parseU32(argv[1], "M");
    const uint32_t n = parseU32(argv[2], "N");
    const uint32_t k = parseU32(argv[3], "K");
    const uint32_t tile_dim = parseU32(argv[4], "ARRAY_DIM");
    const uintptr_t spm_base = parseU32(argv[5], "SPM_BASE");
    const uint32_t spm_size = parseU32(argv[6], "SPM_SIZE");
    const uint64_t a_bytes = paddedElements(m, k, tile_dim, k);
    const uint64_t b_bytes = paddedElements(k, n, k, tile_dim);
    const uint64_t c_bytes =
        paddedElements(m, n, tile_dim, tile_dim) * sizeof(int32_t);
    const uintptr_t a_base = spm_base;
    const uintptr_t b_base = align64(a_base + a_bytes);
    const uintptr_t c_base = align64(b_base + b_bytes);
    if (c_base + c_bytes > spm_base + spm_size) {
        fprintf(stderr, "ATLAS_MATMUL_WORKSPACE_TOO_SMALL required=%llu available=%u\n",
                static_cast<unsigned long long>(c_base + c_bytes - spm_base),
                spm_size);
        return 2;
    }

    const NpuMpuGemmSpmI8Matrix a_matrix =
        npu_mpu_gemm_i8_blocked_matrix(a_base, m, k, tile_dim, k);
    const NpuMpuGemmSpmI8Matrix b_matrix =
        npu_mpu_gemm_i8_blocked_matrix(b_base, k, n, k, tile_dim);
    const NpuMpuGemmSpmI32Matrix c_matrix =
        npu_mpu_gemm_i32_blocked_matrix(c_base, m, n, tile_dim, tile_dim);
    const NpuMpuGemmProblemShape problem = {m, n, k};
    const NpuMpuGemmTileShape tile = {tile_dim, tile_dim, k};
    const NpuFastMatmulLaunchConfig launch = {0U, 0U, 0U, 0U, 0U, 0U, 0U};
    NpuFastMatmulStats stats = {};
    std::vector<int8_t> a_values(static_cast<size_t>(m) * k);
    std::vector<int8_t> b_values(static_cast<size_t>(k) * n);

    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < k; ++col) {
            a_values[static_cast<size_t>(row) * k + col] = aValue(row, col);
        }
    }
    for (uint32_t row = 0; row < k; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            b_values[static_cast<size_t>(row) * n + col] = bValue(row, col);
        }
    }
    npu_mpu_gemm_store_i8_matrix_to_spm(a_values.data(), &a_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(b_values.data(), &b_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&c_matrix, OutputSentinel);

    if (npu_fast_matmul_launch_tiled_spm(
            &a_matrix, &b_matrix, &c_matrix, &problem, &tile, &launch,
            &stats) != 0) {
        fprintf(stderr, "ATLAS_MATMUL_LAUNCH_FAIL\n");
        return 1;
    }
    volatile int32_t *last = reinterpret_cast<volatile int32_t *>(
        npu_mpu_gemm_i32_addr(&c_matrix, m - 1U, n - 1U));
    if (npu_wait_i32_vector_not_value(
            nullptr, last, OutputSentinel, 1U, WaitTimeout) != 0) {
        fprintf(stderr, "ATLAS_MATMUL_WAIT_FAIL\n");
        return 1;
    }

    for (uint32_t sample = 0; sample < 8U; ++sample) {
        const uint32_t row = (sample * 17U) % m;
        const uint32_t col = (sample * 67U) % n;
        const volatile int32_t *actual =
            reinterpret_cast<const volatile int32_t *>(
                npu_mpu_gemm_i32_addr(&c_matrix, row, col));
        const int32_t expected = referenceValue(row, col, k);
        if (*actual != expected) {
            fprintf(stderr,
                    "ATLAS_MATMUL_MISMATCH row=%u col=%u expected=%d actual=%d\n",
                    row, col, expected, *actual);
            return 1;
        }
    }

    printf("ATLAS_MATMUL_WORKLOAD_PASS m=%u n=%u k=%u tiles=%u\n", m, n,
           k, stats.tile_count);
    return 0;
}
