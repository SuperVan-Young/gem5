#ifndef TESTS_GEM5_NPU_UTILS_GOLDEN_VPU_UNARY_H_
#define TESTS_GEM5_NPU_UTILS_GOLDEN_VPU_UNARY_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void npu_golden_vpu_unary_scale_i32(const int32_t *src, int32_t *dst,
                                    uint32_t count, uint32_t repetition,
                                    int32_t scalar);
void npu_golden_vpu_unary_scale_f32(const uint32_t *src_bits, uint32_t *dst_bits,
                                    uint32_t count, uint32_t repetition,
                                    uint32_t scalar_bits);
void npu_golden_vpu_unary_i2f(const int32_t *src, uint32_t *dst_bits,
                              uint32_t count);
void npu_golden_vpu_unary_f2i(const uint32_t *src_bits, int32_t *dst,
                              uint32_t count);
void npu_golden_vpu_unary_sqrt(const uint32_t *src_bits, uint32_t *dst_bits,
                               uint32_t count);
void npu_golden_vpu_unary_exp(const uint32_t *src_bits, uint32_t *dst_bits,
                              uint32_t count);

#ifdef __cplusplus
}
#endif

#endif
