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
    NpuCmd compute_fused;
    NpuCmd drain;
    NpuCmd mvout;
} NpuFastMatmulCmdTemplates;

#define NPU_FAST_MATMUL_PAIR32(lo, hi) \
    ((((uint64_t)(uint32_t)(hi)) << 32) | (uint64_t)(uint32_t)(lo))

#define NPU_FAST_MATMUL_LAUNCH_RAW(template_cmd, buffer_word, spm_addr, \
                                   m_value, n_value, k_value, sns_value) \
    do { \
        uint32_t header_word = (template_cmd).getWord(0U); \
        if ((sns_value) != 0U) { \
            header_word |= (1U << 7); \
        } else { \
            header_word &= ~(1U << 7); \
        } \
        const uint32_t spm_addr_lo = \
            (uint32_t)((uint64_t)(spm_addr) & 0xffffffffULL); \
        const uint32_t spm_addr_hi = (uint32_t)((uint64_t)(spm_addr) >> 32); \
        npuCmdLaunchRawPairsInsn( \
            NPU_FAST_MATMUL_PAIR32(header_word, (template_cmd).getWord(1U)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(2U), \
                                   (template_cmd).getWord(3U)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(4U), \
                                   (buffer_word)), \
            NPU_FAST_MATMUL_PAIR32((m_value), (n_value)), \
            NPU_FAST_MATMUL_PAIR32((k_value), spm_addr_lo), \
            NPU_FAST_MATMUL_PAIR32(spm_addr_hi, \
                                   (template_cmd).getWord(MPU_WORD_STRIDE)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(MPU_WORD_FLAGS), \
                                   (template_cmd).getWord(13U)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(14U), \
                                   (template_cmd).getWord(15U))); \
    } while (0)

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
npu_fast_matmul_set_addr(NpuCmd *cmd, uintptr_t spm_addr)
{
    cmd->setWord(MPU_WORD_SPM_ADDR_LO,
                 (uint32_t)(spm_addr & 0xffffffffULL));
    cmd->setWord(MPU_WORD_SPM_ADDR_HI, (uint32_t)(spm_addr >> 32));
}

static inline void
npu_fast_matmul_set_dims(NpuCmd *cmd, uint32_t m, uint32_t n, uint32_t k)
{
    cmd->setWord(MPU_WORD_M, m);
    cmd->setWord(MPU_WORD_N, n);
    cmd->setWord(MPU_WORD_K, k);
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
    mpu_build_cmd(&templates->compute_fused, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_COMPUTE_FUSED),
                  MPU_BUFFER_RESERVED, 0U, tile->tile_m, tile->tile_n,
                  tile->tile_k, 0U, 0U, config->sync_indicator, 0U);
    templates->compute_fused.setWord(MPU_WORD_FLAGS,
                                     MPU_FLAG_LAST_K_BLOCK);
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
    templates.mvin_a.setWord(MPU_WORD_STRIDE, a_matrix->row_stride_bytes);
    templates.mvin_b.setWord(MPU_WORD_STRIDE, b_matrix->row_stride_bytes);
    templates.mvout.setWord(MPU_WORD_STRIDE, c_matrix->row_stride_bytes);

    if (stats != NULL) {
        *stats = {};
        stats->template_build_count = 5U;
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
            const bool edge_tile =
                rows != tile->tile_m || cols != tile->tile_n ||
                problem->k != tile->tile_k;

            const uint32_t a_buffer =
                config->a_buffer_index ^ buffer_index;
            const uint32_t b_buffer =
                config->b_buffer_index ^ buffer_index;
            const uint32_t c_buffer =
                config->c_buffer_index ^ buffer_index;
            const uint32_t cmd_m =
                edge_tile ? rows :
                    templates.compute_fused.getWord(MPU_WORD_M);
            const uint32_t cmd_n =
                edge_tile ? cols :
                    templates.compute_fused.getWord(MPU_WORD_N);
            const uint32_t cmd_k =
                edge_tile ? problem->k :
                    templates.compute_fused.getWord(MPU_WORD_K);

            NPU_FAST_MATMUL_LAUNCH_RAW(
                templates.mvin_a, mpu_buffer_word(MPU_BUFFER_A, a_buffer),
                a_addr, cmd_m, cmd_n, cmd_k, 0U);
            NPU_FAST_MATMUL_LAUNCH_RAW(
                templates.mvin_b, mpu_buffer_word(MPU_BUFFER_B, b_buffer),
                b_addr, cmd_m, cmd_n, cmd_k, 0U);
            NPU_FAST_MATMUL_LAUNCH_RAW(
                templates.compute_fused, mpu_fused_buffer_word(buffer_index),
                0U, cmd_m, cmd_n, cmd_k, 0U);
            NPU_FAST_MATMUL_LAUNCH_RAW(
                templates.drain, mpu_buffer_word(MPU_BUFFER_C, c_buffer),
                0U, cmd_m, cmd_n, cmd_k, 0U);
            NPU_FAST_MATMUL_LAUNCH_RAW(
                templates.mvout, mpu_buffer_word(MPU_BUFFER_C, c_buffer),
                c_addr, cmd_m, cmd_n, cmd_k, set_completion_sync);

            if (stats != NULL) {
                stats->launched_cmd_count += 5U;
            }
            tile_ordinal += 1U;
        }
    }

    return 0;
}

#endif
