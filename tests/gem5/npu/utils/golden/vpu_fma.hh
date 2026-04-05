#ifndef TESTS_GEM5_NPU_UTILS_GOLDEN_VPU_FMA_H_
#define TESTS_GEM5_NPU_UTILS_GOLDEN_VPU_FMA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void npu_golden_vpu_fma_f32(const uint32_t *lhs_bits,
                            const uint32_t *rhs_bits,
                            const uint32_t *acc_bits, uint32_t *dst_bits,
                            uint32_t count);

#ifdef __cplusplus
}
#endif

#endif
