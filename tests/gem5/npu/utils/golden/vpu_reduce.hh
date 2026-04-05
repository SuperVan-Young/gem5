#ifndef TESTS_GEM5_NPU_UTILS_GOLDEN_VPU_REDUCE_H_
#define TESTS_GEM5_NPU_UTILS_GOLDEN_VPU_REDUCE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void npu_golden_vpu_reduce_sum_i32(const int32_t *src, int32_t *dst,
                                   uint32_t count);
void npu_golden_vpu_reduce_sum_f32(const uint32_t *src_bits,
                                   uint32_t *dst_bits, uint32_t count);
void npu_golden_vpu_reduce_max_f32(const uint32_t *src_bits,
                                   uint32_t *dst_bits, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif
