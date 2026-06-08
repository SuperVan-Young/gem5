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
    NpuCmd fused_matmul;
} NpuFastMatmulCmdTemplates;

#define NPU_FAST_MATMUL_PAIR32(lo, hi) \
    ((((uint64_t)(uint32_t)(hi)) << 32) | (uint64_t)(uint32_t)(lo))

#define NPU_FAST_MATMUL_LAUNCH_CMD(template_cmd, sns_value) \
    do { \
        uint32_t header_word = (template_cmd).getWord(0U); \
        if ((sns_value) != 0U) { \
            header_word |= (1U << 7); \
        } else { \
            header_word &= ~(1U << 7); \
        } \
        npuCmdLaunchRawPairsInsn( \
            NPU_FAST_MATMUL_PAIR32(header_word, (template_cmd).getWord(1U)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(2U), \
                                   (template_cmd).getWord(3U)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(4U), \
                                   (template_cmd).getWord(5U)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(6U), \
                                   (template_cmd).getWord(7U)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(8U), \
                                   (template_cmd).getWord(9U)), \
            NPU_FAST_MATMUL_PAIR32((template_cmd).getWord(10U), \
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

static inline uint32_t
npu_fast_matmul_pack_tile_shape(const NpuMpuGemmTileShape *tile)
{
    return (tile->tile_m & 0xffU) | ((tile->tile_n & 0xffU) << 8) |
           ((tile->tile_k & 0xffU) << 16);
}

static inline uint32_t
npu_fast_matmul_pack_strides(uint32_t a_stride_bytes,
                             uint32_t b_stride_bytes,
                             uint32_t c_stride_bytes)
{
    return ((a_stride_bytes / 16U) & 0x3ffU) |
           (((b_stride_bytes / 16U) & 0x3ffU) << 10) |
           (((c_stride_bytes / 16U) & 0xfffU) << 20);
}

static inline void
npu_fast_matmul_build_cmd_templates(
    const NpuMpuGemmTileShape *tile, const NpuFastMatmulLaunchConfig *config,
    NpuFastMatmulCmdTemplates *templates)
{
    mpu_build_cmd(&templates->fused_matmul, config->device_id,
                  mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_FUSED_MATMUL),
                  MPU_BUFFER_RESERVED, 0U, 0U, 0U, 0U, 0U, 0U,
                  config->sync_indicator, config->set_completion_sync);
    templates->fused_matmul.setWord(
        MPU_WORD_BUFFER, npu_fast_matmul_pack_tile_shape(tile));
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

    if (npu_mpu_gemm_validate_problem(a_matrix, b_matrix, c_matrix, problem,
                                      tile) != 0) {
        return -1;
    }
    if (npu_fast_matmul_validate_launch_config(config) != 0) {
        return -1;
    }

    npu_fast_matmul_build_cmd_templates(tile, config, &templates);
    templates.fused_matmul.setWord(MPU_WORD_M, problem->m);
    templates.fused_matmul.setWord(MPU_WORD_N, problem->n);
    templates.fused_matmul.setWord(MPU_WORD_K, problem->k);
    templates.fused_matmul.setWord(
        MPU_WORD_SPM_ADDR_LO,
        (uint32_t)((uint64_t)a_matrix->base_addr & 0xffffffffULL));
    templates.fused_matmul.setWord(
        MPU_WORD_SPM_ADDR_HI, (uint32_t)((uint64_t)a_matrix->base_addr >> 32));
    templates.fused_matmul.setWord(
        MPU_WORD_STRIDE,
        (uint32_t)((uint64_t)b_matrix->base_addr & 0xffffffffULL));
    templates.fused_matmul.setWord(
        MPU_WORD_FLAGS,
        (uint32_t)((uint64_t)c_matrix->base_addr & 0xffffffffULL));
    templates.fused_matmul.setWord(
        13U, (uint32_t)((uint64_t)b_matrix->base_addr >> 32));
    templates.fused_matmul.setWord(
        14U, (uint32_t)((uint64_t)c_matrix->base_addr >> 32));
    templates.fused_matmul.setWord(
        15U, npu_fast_matmul_pack_strides(a_matrix->row_stride_bytes,
                                          b_matrix->row_stride_bytes,
                                          c_matrix->row_stride_bytes));

    if (stats != NULL) {
        *stats = {};
        stats->template_build_count = 1U;
        stats->tile_count = tile_count;
    }

    if (tile->tile_k != problem->k ||
        (a_matrix->row_stride_bytes % 16U) != 0U ||
        (b_matrix->row_stride_bytes % 16U) != 0U ||
        (c_matrix->row_stride_bytes % 16U) != 0U) {
        printf("FAST_MATMUL_VALIDATE_FAIL=unsupported_fused_shape\n");
        return -1;
    }

    NPU_FAST_MATMUL_LAUNCH_CMD(templates.fused_matmul,
                               config->set_completion_sync);

    if (stats != NULL) {
        stats->launched_cmd_count = 1U;
    }

    return 0;
}

#endif
