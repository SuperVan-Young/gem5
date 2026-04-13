/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "golden/mpu_gemm.hh"
#include "mpu_gemm.hh"
#include "mpu_gemm_layout.hh"
#include "npu_assert.hh"

enum BasicMpuGemmTiledLayout
{
    GEMM_DEVICE_ID = 0U,
    GEMM_M = 4U,
    GEMM_N = 4U,
    GEMM_K = 4U,
    GEMM_TILE_M = 2U,
    GEMM_TILE_N = 2U,
    GEMM_TILE_K = 4U,
    GEMM_SYNC_INDICATOR = 0U,
    GEMM_SET_COMPLETION_SYNC = 0U,
    A0_SPM = 0x60004000UL,
    B0_SPM = 0x60005000UL,
    C0_SPM = 0x60006000UL,
    WAIT_TIMEOUT = 80000000ULL,
};

int
main(int argc, char **argv)
{
    const char *scenario = argc > 1 ? argv[1] : "basic_gemm_tiled";
    const int8_t a_values[GEMM_M * GEMM_K] = {
        1, 2, 3, 4,
        0, 1, 0, 1,
        2, -1, 1, 3,
        4, 0, -2, 2,
    };
    const int8_t b_values[GEMM_K * GEMM_N] = {
        1, 0, 2, 1,
        -1, 3, 1, 0,
        2, 1, 0, -1,
        1, -2, 4, 2,
    };
    int32_t expected[GEMM_M * GEMM_N] = {};
    int32_t actual[GEMM_M * GEMM_N] = {};
    const NpuMpuGemmSpmI8Matrix a_matrix = {
        A0_SPM,
        GEMM_M,
        GEMM_K,
        npu_mpu_gemm_i8_row_stride_bytes(GEMM_K),
    };
    const NpuMpuGemmSpmI8Matrix b_matrix = {
        B0_SPM,
        GEMM_K,
        GEMM_N,
        npu_mpu_gemm_i8_row_stride_bytes(GEMM_N),
    };
    const NpuMpuGemmSpmI32Matrix c_matrix = {
        C0_SPM,
        GEMM_M,
        GEMM_N,
        npu_mpu_gemm_i32_row_stride_bytes(GEMM_N),
    };
    const NpuMpuGemmProblemShape problem = {GEMM_M, GEMM_N, GEMM_K};
    const NpuMpuGemmTileShape tile = {
        GEMM_TILE_M,
        GEMM_TILE_N,
        GEMM_TILE_K,
    };
    const NpuMpuGemmLaunchConfig launch = {
        GEMM_DEVICE_ID,
        GEMM_SYNC_INDICATOR,
        GEMM_SET_COMPLETION_SYNC,
        0U,
        0U,
        0U,
    };

    if (strcmp(scenario, "basic_gemm_tiled") != 0) {
        printf("MPU_UNKNOWN_SCENARIO=%s\n", scenario);
        return 1;
    }

    npu_mpu_gemm_store_i8_matrix_to_spm(a_values, &a_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(b_values, &b_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&c_matrix, -1);
    npu_golden_mpu_gemm_i8_i8_i32(a_values, b_values, expected, GEMM_M,
                                  GEMM_N, GEMM_K, GEMM_K, GEMM_N, GEMM_N);

    if (npu_mpu_gemm_launch_tiled_mn(&a_matrix, &b_matrix, &c_matrix,
                                     &problem, &tile, &launch) != 0) {
        printf("MPU_BASIC_GEMM_TILED_LAUNCH_FAIL\n");
        return 1;
    }

    if (npu_wait_u32_vector_match(
            NULL, (volatile uint32_t *)(uintptr_t)c_matrix.base_addr,
            (const uint32_t *)expected, GEMM_M * GEMM_N,
            WAIT_TIMEOUT) != 0) {
        npu_mpu_gemm_load_i32_matrix_from_spm(&c_matrix, actual);
        printf("MPU_BASIC_GEMM_TILED_FAIL\n");
        npu_expect_u32_vector("MPU_BASIC_GEMM_TILED",
                              (const uint32_t *)expected,
                              (const uint32_t *)actual,
                              GEMM_M * GEMM_N);
        return 1;
    }

    printf("MPU_SCENARIO_PASS=basic_gemm_tiled\n");
    return 0;
}
