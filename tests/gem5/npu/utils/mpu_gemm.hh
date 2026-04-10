#ifndef TESTS_GEM5_NPU_UTILS_MPU_GEMM_H_
#define TESTS_GEM5_NPU_UTILS_MPU_GEMM_H_

#include <stdint.h>

#include <cstdio>

#include "cmd/mpu.hh"
#include "mpu_gemm_layout.hh"

typedef struct
{
    uint32_t device_id;
    uint32_t sync_indicator;
    uint32_t set_completion_sync;
    uint32_t a_buffer_index;
    uint32_t b_buffer_index;
    uint32_t c_buffer_index;
} NpuMpuGemmLaunchConfig;

static inline int
npu_mpu_gemm_validate_launch_config(const NpuMpuGemmLaunchConfig *config)
{
    if (config == NULL) {
        printf("MPU_GEMM_VALIDATE_FAIL=null_launch_config\n");
        return -1;
    }
    if (config->a_buffer_index > 1U || config->b_buffer_index > 1U ||
        config->c_buffer_index > 1U) {
        printf("MPU_GEMM_VALIDATE_FAIL=invalid_buffer_index\n");
        return -1;
    }
    return 0;
}

static inline int
npu_mpu_gemm_launch_single_tile(const NpuMpuGemmSpmI8Matrix *a_matrix,
                                const NpuMpuGemmSpmI8Matrix *b_matrix,
                                const NpuMpuGemmSpmI32Matrix *c_matrix,
                                const NpuMpuGemmProblemShape *problem,
                                const NpuMpuGemmTileShape *tile,
                                const NpuMpuGemmLaunchConfig *config,
                                uint32_t row0, uint32_t col0)
{
    uint32_t tile_rows = 0U;
    uint32_t tile_cols = 0U;
    const uint32_t tile_k = tile->tile_k;
    uintptr_t a_addr = 0U;
    uintptr_t b_addr = 0U;
    uintptr_t c_addr = 0U;

    if (npu_mpu_gemm_validate_problem(a_matrix, b_matrix, c_matrix, problem,
                                      tile) != 0) {
        return -1;
    }
    if (npu_mpu_gemm_validate_launch_config(config) != 0) {
        return -1;
    }
    if (row0 >= problem->m || col0 >= problem->n) {
        printf("MPU_GEMM_VALIDATE_FAIL=tile_origin_out_of_range\n");
        return -1;
    }

    tile_rows = npu_mpu_gemm_tile_rows(problem, tile, row0);
    tile_cols = npu_mpu_gemm_tile_cols(problem, tile, col0);
    a_addr = npu_mpu_gemm_a_tile_addr(a_matrix, row0);
    b_addr = npu_mpu_gemm_b_tile_addr(b_matrix, col0);
    c_addr = npu_mpu_gemm_c_tile_addr(c_matrix, row0, col0);

    mpu_launch_cmd(config->device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVIN), MPU_BUFFER_A,
                   config->a_buffer_index, tile_rows, tile_cols, tile_k,
                   a_addr, a_matrix->row_stride_bytes, config->sync_indicator,
                   0U);
    mpu_launch_cmd(config->device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVIN), MPU_BUFFER_B,
                   config->b_buffer_index, tile_rows, tile_cols, tile_k,
                   b_addr, b_matrix->row_stride_bytes, config->sync_indicator,
                   0U);
    mpu_launch_cmd(config->device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_LOAD), MPU_BUFFER_A,
                   config->a_buffer_index, tile_rows, tile_cols, tile_k, 0U,
                   0U, config->sync_indicator, 0U);
    mpu_launch_cmd(config->device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_LOAD), MPU_BUFFER_B,
                   config->b_buffer_index, tile_rows, tile_cols, tile_k, 0U,
                   0U, config->sync_indicator, 0U);
    mpu_launch_cmd(config->device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_COMPUTE),
                   MPU_BUFFER_RESERVED, 0U, tile_rows, tile_cols, tile_k, 0U,
                   0U, config->sync_indicator, 0U);
    mpu_launch_cmd(config->device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_DRAIN),
                   MPU_BUFFER_C, config->c_buffer_index, tile_rows, tile_cols,
                   tile_k, 0U, 0U, config->sync_indicator, 0U);
    mpu_launch_cmd(config->device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVOUT),
                   MPU_BUFFER_C, config->c_buffer_index, tile_rows, tile_cols,
                   tile_k, c_addr, c_matrix->row_stride_bytes,
                   config->sync_indicator, config->set_completion_sync);

    return 0;
}

static inline int
npu_mpu_gemm_launch_tiled_mn(const NpuMpuGemmSpmI8Matrix *a_matrix,
                             const NpuMpuGemmSpmI8Matrix *b_matrix,
                             const NpuMpuGemmSpmI32Matrix *c_matrix,
                             const NpuMpuGemmProblemShape *problem,
                             const NpuMpuGemmTileShape *tile,
                             const NpuMpuGemmLaunchConfig *config)
{
    if (npu_mpu_gemm_validate_problem(a_matrix, b_matrix, c_matrix, problem,
                                      tile) != 0) {
        return -1;
    }
    if (npu_mpu_gemm_validate_launch_config(config) != 0) {
        return -1;
    }

    for (uint32_t row0 = 0U; row0 < problem->m; row0 += tile->tile_m) {
        for (uint32_t col0 = 0U; col0 < problem->n; col0 += tile->tile_n) {
            if (npu_mpu_gemm_launch_single_tile(a_matrix, b_matrix, c_matrix,
                                                problem, tile, config, row0,
                                                col0) != 0) {
                return -1;
            }
        }
    }

    return 0;
}

#endif
