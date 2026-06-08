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
    uint32_t block_rows;
    uint32_t block_cols;
} NpuMpuGemmSpmI8Matrix;

typedef struct
{
    uintptr_t base_addr;
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride_bytes;
    uint32_t block_rows;
    uint32_t block_cols;
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

static inline uint32_t
npu_mpu_gemm_div_ceil_u32(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

static inline bool
npu_mpu_gemm_i8_is_blocked(const NpuMpuGemmSpmI8Matrix *matrix)
{
    return matrix->block_rows != 0U && matrix->block_cols != 0U;
}

static inline bool
npu_mpu_gemm_i32_is_blocked(const NpuMpuGemmSpmI32Matrix *matrix)
{
    return matrix->block_rows != 0U && matrix->block_cols != 0U;
}

static inline NpuMpuGemmSpmI8Matrix
npu_mpu_gemm_i8_blocked_matrix(uintptr_t base_addr, uint32_t rows,
                               uint32_t cols, uint32_t block_rows,
                               uint32_t block_cols)
{
    const NpuMpuGemmSpmI8Matrix matrix = {
        base_addr,
        rows,
        cols,
        npu_mpu_gemm_i8_row_stride_bytes(block_cols),
        block_rows,
        block_cols,
    };
    return matrix;
}

static inline NpuMpuGemmSpmI32Matrix
npu_mpu_gemm_i32_blocked_matrix(uintptr_t base_addr, uint32_t rows,
                                uint32_t cols, uint32_t block_rows,
                                uint32_t block_cols)
{
    const NpuMpuGemmSpmI32Matrix matrix = {
        base_addr,
        rows,
        cols,
        npu_mpu_gemm_i32_row_stride_bytes(block_cols),
        block_rows,
        block_cols,
    };
    return matrix;
}

static inline uintptr_t
npu_mpu_gemm_i8_addr(const NpuMpuGemmSpmI8Matrix *matrix, uint32_t row,
                     uint32_t col)
{
    if (!npu_mpu_gemm_i8_is_blocked(matrix)) {
        return matrix->base_addr +
               (uintptr_t)row * matrix->row_stride_bytes + col;
    }

    const uint32_t blocks_per_row =
        npu_mpu_gemm_div_ceil_u32(matrix->cols, matrix->block_cols);
    const uint32_t block_row = row / matrix->block_rows;
    const uint32_t block_col = col / matrix->block_cols;
    const uint32_t inner_row = row % matrix->block_rows;
    const uint32_t inner_col = col % matrix->block_cols;
    const uintptr_t block_elems =
        (uintptr_t)matrix->block_rows * matrix->block_cols;
    const uintptr_t block_index =
        (uintptr_t)block_row * blocks_per_row + block_col;
    return matrix->base_addr + block_index * block_elems +
           (uintptr_t)inner_row * matrix->block_cols + inner_col;
}

static inline uintptr_t
npu_mpu_gemm_i32_addr(const NpuMpuGemmSpmI32Matrix *matrix, uint32_t row,
                      uint32_t col)
{
    if (!npu_mpu_gemm_i32_is_blocked(matrix)) {
        return matrix->base_addr +
               (uintptr_t)row * matrix->row_stride_bytes +
               (uintptr_t)col * sizeof(int32_t);
    }

    const uint32_t blocks_per_row =
        npu_mpu_gemm_div_ceil_u32(matrix->cols, matrix->block_cols);
    const uint32_t block_row = row / matrix->block_rows;
    const uint32_t block_col = col / matrix->block_cols;
    const uint32_t inner_row = row % matrix->block_rows;
    const uint32_t inner_col = col % matrix->block_cols;
    const uintptr_t block_elems =
        (uintptr_t)matrix->block_rows * matrix->block_cols;
    const uintptr_t block_index =
        (uintptr_t)block_row * blocks_per_row + block_col;
    return matrix->base_addr +
           (block_index * block_elems +
            (uintptr_t)inner_row * matrix->block_cols + inner_col) *
               sizeof(int32_t);
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

    if (!npu_mpu_gemm_i8_is_blocked(a_matrix) &&
        a_matrix->row_stride_bytes !=
            npu_mpu_gemm_i8_row_stride_bytes(a_matrix->cols)) {
        printf("MPU_GEMM_VALIDATE_FAIL=a_not_dense_row_major\n");
        return -1;
    }
    if (!npu_mpu_gemm_i8_is_blocked(b_matrix) &&
        b_matrix->row_stride_bytes !=
            npu_mpu_gemm_i8_row_stride_bytes(b_matrix->cols)) {
        printf("MPU_GEMM_VALIDATE_FAIL=b_not_dense_row_major\n");
        return -1;
    }
    if (!npu_mpu_gemm_i32_is_blocked(c_matrix) &&
        c_matrix->row_stride_bytes !=
            npu_mpu_gemm_i32_row_stride_bytes(c_matrix->cols)) {
        printf("MPU_GEMM_VALIDATE_FAIL=c_not_dense_row_major\n");
        return -1;
    }
    if (npu_mpu_gemm_i8_is_blocked(a_matrix) &&
        (a_matrix->block_rows != tile->tile_m ||
         a_matrix->block_cols != tile->tile_k ||
         a_matrix->row_stride_bytes !=
             npu_mpu_gemm_i8_row_stride_bytes(tile->tile_k))) {
        printf("MPU_GEMM_VALIDATE_FAIL=a_block_layout_mismatch\n");
        return -1;
    }
    if (npu_mpu_gemm_i8_is_blocked(b_matrix) &&
        (b_matrix->block_rows != tile->tile_k ||
         b_matrix->block_cols != tile->tile_n ||
         b_matrix->row_stride_bytes !=
             npu_mpu_gemm_i8_row_stride_bytes(tile->tile_n))) {
        printf("MPU_GEMM_VALIDATE_FAIL=b_block_layout_mismatch\n");
        return -1;
    }
    if (npu_mpu_gemm_i32_is_blocked(c_matrix) &&
        (c_matrix->block_rows != tile->tile_m ||
         c_matrix->block_cols != tile->tile_n ||
         c_matrix->row_stride_bytes !=
             npu_mpu_gemm_i32_row_stride_bytes(tile->tile_n))) {
        printf("MPU_GEMM_VALIDATE_FAIL=c_block_layout_mismatch\n");
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
    return npu_mpu_gemm_i8_addr(a_matrix, row0, 0U);
}

static inline uintptr_t
npu_mpu_gemm_b_tile_addr(const NpuMpuGemmSpmI8Matrix *b_matrix, uint32_t col0)
{
    return npu_mpu_gemm_i8_addr(b_matrix, 0U, col0);
}

static inline uintptr_t
npu_mpu_gemm_c_tile_addr(const NpuMpuGemmSpmI32Matrix *c_matrix, uint32_t row0,
                         uint32_t col0)
{
    return npu_mpu_gemm_i32_addr(c_matrix, row0, col0);
}

static inline void
npu_mpu_gemm_store_i8_matrix_to_spm(const int8_t *src,
                                    const NpuMpuGemmSpmI8Matrix *dst)
{
    for (uint32_t row = 0U; row < dst->rows; ++row) {
        for (uint32_t col = 0U; col < dst->cols; ++col) {
            volatile uint8_t *dst_elem = (volatile uint8_t *)(uintptr_t)
                npu_mpu_gemm_i8_addr(dst, row, col);
            *dst_elem = (uint8_t)src[row * dst->cols + col];
        }
    }
}

static inline void
npu_mpu_gemm_fill_i32_matrix_in_spm(const NpuMpuGemmSpmI32Matrix *dst,
                                    int32_t value)
{
    for (uint32_t row = 0U; row < dst->rows; ++row) {
        for (uint32_t col = 0U; col < dst->cols; ++col) {
            volatile int32_t *dst_elem = (volatile int32_t *)(uintptr_t)
                npu_mpu_gemm_i32_addr(dst, row, col);
            *dst_elem = value;
        }
    }
}

static inline void
npu_mpu_gemm_load_i32_matrix_from_spm(const NpuMpuGemmSpmI32Matrix *src,
                                      int32_t *dst)
{
    for (uint32_t row = 0U; row < src->rows; ++row) {
        for (uint32_t col = 0U; col < src->cols; ++col) {
            const volatile int32_t *src_elem =
                (const volatile int32_t *)(uintptr_t)
                    npu_mpu_gemm_i32_addr(src, row, col);
            dst[row * src->cols + col] = *src_elem;
        }
    }
}

#endif
