#ifndef TESTS_GEM5_NPU_UTILS_GOLDEN_VPU_ELEMWISE_H_
#define TESTS_GEM5_NPU_UTILS_GOLDEN_VPU_ELEMWISE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void npu_golden_vpu_elemwise_i32(uint32_t op_code, const uint32_t *lhs,
                                 const uint32_t *rhs, uint32_t *dst,
                                 uint32_t count);
void npu_golden_vpu_elemwise_f32(uint32_t op_code, const uint32_t *lhs_bits,
                                 const uint32_t *rhs_bits, uint32_t *dst_bits,
                                 uint32_t count);

#ifdef __cplusplus
}
#endif

#endif
