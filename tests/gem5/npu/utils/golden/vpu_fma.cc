#include "vpu_fma.hh"

#include <math.h>

#include "../npu_assert.hh"

void
npu_golden_vpu_fma_f32(const uint32_t *lhs_bits, const uint32_t *rhs_bits,
                       const uint32_t *acc_bits, uint32_t *dst_bits,
                       uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        const float lhs = npu_bits_to_float(lhs_bits[idx]);
        const float rhs = npu_bits_to_float(rhs_bits[idx]);
        const float acc = npu_bits_to_float(acc_bits[idx]);
        dst_bits[idx] = npu_float_to_bits(fmaf(lhs, rhs, acc));
    }
}
