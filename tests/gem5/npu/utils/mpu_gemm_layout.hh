#ifndef TESTS_GEM5_NPU_UTILS_MPU_GEMM_LAYOUT_H_
#define TESTS_GEM5_NPU_UTILS_MPU_GEMM_LAYOUT_H_

#include <stdint.h>

#include <cstdio>

typedef struct
{
    uintptr_t base_addr;
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride_bytes;
} NpuMpuGemmSpmI8Matrix;

typedef struct
{
    uintptr_t base_addr;
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride_bytes;
} NpuMpuGemmSpmI32Matrix;

typedef struct
{
    uint32_t m;
    uint32_t n;
    uint32_t k;
} NpuMpuGemmProblemShape;

typedef struct
{
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tile_k;
} NpuMpuGemmTileShape;

static inline uint32_t
npu_mpu_gemm_i8_row_stride_bytes(uint32_t cols)
{
    return cols;
}

static inline uint32_t
npu_mpu_gemm_i32_row_stride_bytes(uint32_t cols)
{
    return cols * sizeof(int32_t);
}

static inline int
npu_mpu_gemm_validate_problem(const NpuMpuGemmSpmI8Matrix *a_matrix,
                              const NpuMpuGemmSpmI8Matrix *b_matrix,
                              const NpuMpuGemmSpmI32Matrix *c_matrix,
                              const NpuMpuGemmProblemShape *problem,
                              const NpuMpuGemmTileShape *tile)
{
    if (a_matrix == NULL || b_matrix == NULL || c_matrix == NULL ||
        problem == NULL || tile == NULL) {
        printf("MPU_GEMM_VALIDATE_FAIL=null_argument\n");
        return -1;
    }

    if (problem->m == 0U || problem->n == 0U || problem->k == 0U) {
        printf("MPU_GEMM_VALIDATE_FAIL=zero_problem_dimension\n");
        return -1;
    }
    if (tile->tile_m == 0U || tile->tile_n == 0U || tile->tile_k == 0U) {
        printf("MPU_GEMM_VALIDATE_FAIL=zero_tile_dimension\n");
        return -1;
    }

    if (a_matrix->rows != problem->m || a_matrix->cols != problem->k) {
        printf("MPU_GEMM_VALIDATE_FAIL=a_shape_mismatch\n");
        return -1;
    }
    if (b_matrix->rows != problem->k || b_matrix->cols != problem->n) {
        printf("MPU_GEMM_VALIDATE_FAIL=b_shape_mismatch\n");
        return -1;
    }
    if (c_matrix->rows != problem->m || c_matrix->cols != problem->n) {
        printf("MPU_GEMM_VALIDATE_FAIL=c_shape_mismatch\n");
        return -1;
    }

    if (a_matrix->row_stride_bytes !=
        npu_mpu_gemm_i8_row_stride_bytes(a_matrix->cols)) {
        printf("MPU_GEMM_VALIDATE_FAIL=a_not_dense_row_major\n");
        return -1;
    }
    if (b_matrix->row_stride_bytes !=
        npu_mpu_gemm_i8_row_stride_bytes(b_matrix->cols)) {
        printf("MPU_GEMM_VALIDATE_FAIL=b_not_dense_row_major\n");
        return -1;
    }
    if (c_matrix->row_stride_bytes !=
        npu_mpu_gemm_i32_row_stride_bytes(c_matrix->cols)) {
        printf("MPU_GEMM_VALIDATE_FAIL=c_not_dense_row_major\n");
        return -1;
    }

    if (tile->tile_m > problem->m || tile->tile_n > problem->n) {
        printf("MPU_GEMM_VALIDATE_FAIL=tile_exceeds_problem\n");
        return -1;
    }
    if (tile->tile_k != problem->k) {
        printf("MPU_GEMM_VALIDATE_FAIL=split_k_not_supported\n");
        return -1;
    }

    return 0;
}

static inline uint32_t
npu_mpu_gemm_tile_rows(const NpuMpuGemmProblemShape *problem,
                       const NpuMpuGemmTileShape *tile, uint32_t row0)
{
    const uint32_t remaining = problem->m - row0;
    return remaining < tile->tile_m ? remaining : tile->tile_m;
}

static inline uint32_t
npu_mpu_gemm_tile_cols(const NpuMpuGemmProblemShape *problem,
                       const NpuMpuGemmTileShape *tile, uint32_t col0)
{
    const uint32_t remaining = problem->n - col0;
    return remaining < tile->tile_n ? remaining : tile->tile_n;
}

static inline uintptr_t
npu_mpu_gemm_a_tile_addr(const NpuMpuGemmSpmI8Matrix *a_matrix, uint32_t row0)
{
    return a_matrix->base_addr +
           (uintptr_t)row0 * (uintptr_t)a_matrix->row_stride_bytes;
}

static inline uintptr_t
npu_mpu_gemm_b_tile_addr(const NpuMpuGemmSpmI8Matrix *b_matrix, uint32_t col0)
{
    return b_matrix->base_addr + (uintptr_t)col0;
}

static inline uintptr_t
npu_mpu_gemm_c_tile_addr(const NpuMpuGemmSpmI32Matrix *c_matrix, uint32_t row0,
                         uint32_t col0)
{
    return c_matrix->base_addr +
           (uintptr_t)row0 * (uintptr_t)c_matrix->row_stride_bytes +
           (uintptr_t)col0 * sizeof(int32_t);
}

static inline void
npu_mpu_gemm_store_i8_matrix_to_spm(const int8_t *src,
                                    const NpuMpuGemmSpmI8Matrix *dst)
{
    for (uint32_t row = 0U; row < dst->rows; ++row) {
        volatile uint8_t *dst_row = (volatile uint8_t *)(uintptr_t)(
            dst->base_addr + (uintptr_t)row * dst->row_stride_bytes);
        for (uint32_t col = 0U; col < dst->cols; ++col) {
            dst_row[col] = (uint8_t)src[row * dst->cols + col];
        }
    }
}

static inline void
npu_mpu_gemm_fill_i32_matrix_in_spm(const NpuMpuGemmSpmI32Matrix *dst,
                                    int32_t value)
{
    for (uint32_t row = 0U; row < dst->rows; ++row) {
        volatile int32_t *dst_row = (volatile int32_t *)(uintptr_t)(
            dst->base_addr + (uintptr_t)row * dst->row_stride_bytes);
        for (uint32_t col = 0U; col < dst->cols; ++col) {
            dst_row[col] = value;
        }
    }
}

static inline void
npu_mpu_gemm_load_i32_matrix_from_spm(const NpuMpuGemmSpmI32Matrix *src,
                                      int32_t *dst)
{
    for (uint32_t row = 0U; row < src->rows; ++row) {
        const uintptr_t row_addr =
            src->base_addr + (uintptr_t)row * src->row_stride_bytes;
        const volatile int32_t *src_row =
            (const volatile int32_t *)row_addr;
        for (uint32_t col = 0U; col < src->cols; ++col) {
            dst[row * src->cols + col] = src_row[col];
        }
    }
}

#endif
