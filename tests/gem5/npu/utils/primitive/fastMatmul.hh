#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_FASTMATMUL_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_FASTMATMUL_HH_

#include <stdint.h>

#include <cstdio>

#include "../cmd/mpu.hh"
#include "../mpu_gemm_layout.hh"

typedef struct
{
    uint32_t device_id;
    uint32_t sync_indicator;
    uint32_t set_completion_sync;
    uint32_t a_buffer_index;
    uint32_t b_buffer_index;
    uint32_t c_buffer_index;
    uint64_t port_base;
} NpuFastMatmulLaunchConfig;

typedef struct
{
    uint32_t template_build_count;
    uint32_t launched_cmd_count;
    uint32_t tile_count;
} NpuFastMatmulStats;

typedef struct
{
    NpuCmd mvin_a;
    NpuCmd mvin_b;
    NpuCmd load_a;
    NpuCmd load_b;
    NpuCmd compute;
    NpuCmd drain;
    NpuCmd mvout;
} NpuFastMatmulCmdTemplates;

static inline int
npu_fast_matmul_validate_launch_config(
    const NpuFastMatmulLaunchConfig *config)
{
    if (config == NULL) {
        printf("FAST_MATMUL_VALIDATE_FAIL=null_launch_config\n");
        return -1;
    }
    if (config->a_buffer_index > 1U || config->b_buffer_index > 1U ||
        config->c_buffer_index > 1U) {
        printf("FAST_MATMUL_VALIDATE_FAIL=invalid_buffer_index\n");
        return -1;
    }
    return 0;
}

static inline void
npu_fast_matmul_patch_cmd(NpuCmd *cmd, uint32_t m, uint32_t n, uint32_t k,
                          uintptr_t spm_addr, uint32_t stride_bytes,
                          uint32_t sync_indicator,
                          uint32_t set_completion_sync)
{
    cmd->setSyncIndicator(sync_indicator);
    cmd->setSetIndicatorSns(set_completion_sync ? 1U : 0U);
    cmd->setWord(MPU_WORD_M, m);
    cmd->setWord(MPU_WORD_N, n);
    cmd->setWord(MPU_WORD_K, k);
    cmd->setWord(MPU_WORD_SPM_ADDR_LO, (uint32_t)(spm_addr & 0xffffffffULL));
    cmd->setWord(MPU_WORD_SPM_ADDR_HI, (uint32_t)(spm_addr >> 32));
    cmd->setWord(MPU_WORD_STRIDE, stride_bytes);
}

static inline void
npu_fast_matmul_patch_buffer(NpuCmd *cmd, uint32_t buffer_kind,
                             uint32_t buffer_index)
{
    cmd->setWord(MPU_WORD_BUFFER,
                 mpu_buffer_word(buffer_kind, buffer_index & 0x1U));
}

static inline void
npu_fast_matmul_build_cmd_templates(
    const NpuMpuGemmTileShape *tile, const NpuFastMatmulLaunchConfig *config,
    NpuFastMatmulCmdTemplates *templates)
{
    mpu_build_cmd(&templates->mvin_a, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVIN),
                  MPU_BUFFER_A, config->a_buffer_index, tile->tile_m,
                  tile->tile_n, tile->tile_k, 0U, 0U,
                  config->sync_indicator, 0U);
    mpu_build_cmd(&templates->mvin_b, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVIN),
                  MPU_BUFFER_B, config->b_buffer_index, tile->tile_m,
                  tile->tile_n, tile->tile_k, 0U, 0U,
                  config->sync_indicator, 0U);
    mpu_build_cmd(&templates->load_a, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_LOAD), MPU_BUFFER_A,
                  config->a_buffer_index, tile->tile_m, tile->tile_n,
                  tile->tile_k, 0U, 0U, config->sync_indicator, 0U);
    mpu_build_cmd(&templates->load_b, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_LOAD), MPU_BUFFER_B,
                  config->b_buffer_index, tile->tile_m, tile->tile_n,
                  tile->tile_k, 0U, 0U, config->sync_indicator, 0U);
    mpu_build_cmd(&templates->compute, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_COMPUTE),
                  MPU_BUFFER_RESERVED, 0U, tile->tile_m, tile->tile_n,
                  tile->tile_k, 0U, 0U, config->sync_indicator, 0U);
    mpu_build_cmd(&templates->drain, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_DRAIN), MPU_BUFFER_C,
                  config->c_buffer_index, tile->tile_m, tile->tile_n,
                  tile->tile_k, 0U, 0U, config->sync_indicator, 0U);
    mpu_build_cmd(&templates->mvout, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVOUT), MPU_BUFFER_C,
                  config->c_buffer_index, tile->tile_m, tile->tile_n,
                  tile->tile_k, 0U, 0U, config->sync_indicator,
                  config->set_completion_sync);
}

static inline int
npu_fast_matmul_launch_tiled_spm(const NpuMpuGemmSpmI8Matrix *a_matrix,
                                 const NpuMpuGemmSpmI8Matrix *b_matrix,
                                 const NpuMpuGemmSpmI32Matrix *c_matrix,
                                 const NpuMpuGemmProblemShape *problem,
                                 const NpuMpuGemmTileShape *tile,
                                 const NpuFastMatmulLaunchConfig *config,
                                 NpuFastMatmulStats *stats)
{
    NpuFastMatmulCmdTemplates templates = {};
    const uint64_t port_base =
        config->port_base == 0U ? NPU_CMD_PORT_BASE : config->port_base;
    const uint32_t tile_rows = (problem->m + tile->tile_m - 1U) / tile->tile_m;
    const uint32_t tile_cols = (problem->n + tile->tile_n - 1U) / tile->tile_n;
    const uint32_t tile_count = tile_rows * tile_cols;
    uint32_t tile_ordinal = 0U;

    if (npu_mpu_gemm_validate_problem(a_matrix, b_matrix, c_matrix, problem,
                                      tile) != 0) {
        return -1;
    }
    if (npu_fast_matmul_validate_launch_config(config) != 0) {
        return -1;
    }

    npu_fast_matmul_build_cmd_templates(tile, config, &templates);

    if (stats != NULL) {
        *stats = {};
        stats->template_build_count = 7U;
        stats->tile_count = tile_count;
    }

    for (uint32_t row0 = 0U; row0 < problem->m; row0 += tile->tile_m) {
        const uint32_t rows = npu_mpu_gemm_tile_rows(problem, tile, row0);

        for (uint32_t col0 = 0U; col0 < problem->n; col0 += tile->tile_n) {
            const uint32_t cols = npu_mpu_gemm_tile_cols(problem, tile, col0);
            const uintptr_t a_addr = npu_mpu_gemm_a_tile_addr(a_matrix, row0);
            const uintptr_t b_addr = npu_mpu_gemm_b_tile_addr(b_matrix, col0);
            const uintptr_t c_addr =
                npu_mpu_gemm_c_tile_addr(c_matrix, row0, col0);
            const uint32_t buffer_index = tile_ordinal & 0x1U;
            const uint32_t set_completion_sync =
                (config->set_completion_sync != 0U &&
                 tile_ordinal == tile_count - 1U) ?
                    1U :
                    0U;

            npu_fast_matmul_patch_buffer(&templates.mvin_a, MPU_BUFFER_A,
                                         config->a_buffer_index ^
                                             buffer_index);
            npu_fast_matmul_patch_buffer(&templates.mvin_b, MPU_BUFFER_B,
                                         config->b_buffer_index ^
                                             buffer_index);
            npu_fast_matmul_patch_buffer(&templates.load_a, MPU_BUFFER_A,
                                         config->a_buffer_index ^
                                             buffer_index);
            npu_fast_matmul_patch_buffer(&templates.load_b, MPU_BUFFER_B,
                                         config->b_buffer_index ^
                                             buffer_index);
            npu_fast_matmul_patch_buffer(&templates.drain, MPU_BUFFER_C,
                                         config->c_buffer_index ^
                                             buffer_index);
            npu_fast_matmul_patch_buffer(&templates.mvout, MPU_BUFFER_C,
                                         config->c_buffer_index ^
                                             buffer_index);
            npu_fast_matmul_patch_cmd(&templates.mvin_a, rows, cols,
                                      problem->k, a_addr,
                                      a_matrix->row_stride_bytes,
                                      config->sync_indicator, 0U);
            npu_fast_matmul_patch_cmd(&templates.mvin_b, rows, cols,
                                      problem->k, b_addr,
                                      b_matrix->row_stride_bytes,
                                      config->sync_indicator, 0U);
            npu_fast_matmul_patch_cmd(&templates.load_a, rows, cols,
                                      problem->k, 0U, 0U,
                                      config->sync_indicator, 0U);
            npu_fast_matmul_patch_cmd(&templates.load_b, rows, cols,
                                      problem->k, 0U, 0U,
                                      config->sync_indicator, 0U);
            npu_fast_matmul_patch_cmd(&templates.compute, rows, cols,
                                      problem->k, 0U, 0U,
                                      config->sync_indicator, 0U);
            npu_fast_matmul_patch_cmd(&templates.drain, rows, cols,
                                      problem->k, 0U, 0U,
                                      config->sync_indicator, 0U);
            npu_fast_matmul_patch_cmd(&templates.mvout, rows, cols,
                                      problem->k, c_addr,
                                      c_matrix->row_stride_bytes,
                                      config->sync_indicator,
                                      set_completion_sync);

            templates.mvin_a.launchCmdAt(port_base);
            templates.mvin_b.launchCmdAt(port_base);
            templates.load_a.launchCmdAt(port_base);
            templates.load_b.launchCmdAt(port_base);
            templates.compute.launchCmdAt(port_base);
            templates.drain.launchCmdAt(port_base);
            templates.mvout.launchCmdAt(port_base);

            if (stats != NULL) {
                stats->launched_cmd_count += 7U;
            }
            tile_ordinal += 1U;
        }
    }

    return 0;
}

#endif
