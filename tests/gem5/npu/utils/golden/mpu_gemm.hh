#ifndef TESTS_GEM5_NPU_UTILS_GOLDEN_MPU_GEMM_H_
#define TESTS_GEM5_NPU_UTILS_GOLDEN_MPU_GEMM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void npu_golden_mpu_gemm_i8_i8_i32(const int8_t *a_data,
                                   const int8_t *b_data,
                                   int32_t *c_data, uint32_t m, uint32_t n,
                                   uint32_t k, uint32_t a_row_stride_elems,
                                   uint32_t b_row_stride_elems,
                                   uint32_t c_row_stride_elems);

#ifdef __cplusplus
}
#endif

#endif
