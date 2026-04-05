#include "vpu_reduce.hh"

#include <math.h>
#include <stdint.h>

#include "../npu_assert.hh"

void
npu_golden_vpu_reduce_sum_i32(const int32_t *src, int32_t *dst, uint32_t count)
{
    int32_t sum = 0;

    for (uint32_t idx = 0U; idx < count; ++idx) {
        sum += src[idx];
    }

    *dst = sum;
}

void
npu_golden_vpu_reduce_sum_f32(const uint32_t *src_bits, uint32_t *dst_bits,
                              uint32_t count)
{
    float sum = 0.0f;

    for (uint32_t idx = 0U; idx < count; ++idx) {
        sum += npu_bits_to_float(src_bits[idx]);
    }

    *dst_bits = npu_float_to_bits(sum);
}

void
npu_golden_vpu_reduce_max_f32(const uint32_t *src_bits, uint32_t *dst_bits,
                              uint32_t count)
{
    float current_max = -INFINITY;

    for (uint32_t idx = 0U; idx < count; ++idx) {
        current_max = fmaxf(current_max, npu_bits_to_float(src_bits[idx]));
    }

    *dst_bits = npu_float_to_bits(current_max);
}
