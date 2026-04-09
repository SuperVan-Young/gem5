#ifndef TESTS_GEM5_NPU_UTILS_GOLDEN_LLM_F32_HH_
#define TESTS_GEM5_NPU_UTILS_GOLDEN_LLM_F32_HH_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void npu_golden_rmsnorm_lastdim_f32(const uint32_t *src_bits,
                                    const uint32_t *weight_bits, uint32_t rows,
                                    uint32_t cols, float epsilon,
                                    uint32_t *dst_bits);
void npu_golden_softmax_lastdim_f32(const uint32_t *src_bits, uint32_t rows,
                                    uint32_t cols, uint32_t *dst_bits);
void npu_golden_swiglu_lastdim_f32(const uint32_t *gate_bits,
                                   const uint32_t *value_bits, uint32_t rows,
                                   uint32_t cols, uint32_t *dst_bits);

#ifdef __cplusplus
}
#endif

#endif
