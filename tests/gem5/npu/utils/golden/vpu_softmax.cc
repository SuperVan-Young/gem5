#include "vpu_softmax.hh"

#include <math.h>

#include "../npu_assert.hh"

void
npu_golden_vpu_softmax_f32(const uint32_t *src_bits, uint32_t *dst_bits,
                           uint32_t count)
{
    float max_value = -INFINITY;
    float sum = 0.0f;

    for (uint32_t idx = 0U; idx < count; ++idx) {
        max_value = fmaxf(max_value, npu_bits_to_float(src_bits[idx]));
    }

    for (uint32_t idx = 0U; idx < count; ++idx) {
        sum += expf(npu_bits_to_float(src_bits[idx]) - max_value);
    }

    for (uint32_t idx = 0U; idx < count; ++idx) {
        const float value =
            expf(npu_bits_to_float(src_bits[idx]) - max_value) / sum;
        dst_bits[idx] = npu_float_to_bits(value);
    }
}
