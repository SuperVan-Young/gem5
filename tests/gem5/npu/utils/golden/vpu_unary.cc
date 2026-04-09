#include "vpu_unary.hh"

#include <math.h>

#include "../npu_assert.hh"

void
npu_golden_vpu_unary_scale_i32(const int32_t *src, int32_t *dst,
                               uint32_t count, uint32_t repetition,
                               int32_t scalar)
{
    (void)repetition;

    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = src[idx] * scalar;
    }
}

void
npu_golden_vpu_unary_scale_f32(const uint32_t *src_bits, uint32_t *dst_bits,
                               uint32_t count, uint32_t repetition,
                               uint32_t scalar_bits)
{
    const float scalar = npu_bits_to_float(scalar_bits);
    (void)repetition;

    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst_bits[idx] =
            npu_float_to_bits(npu_bits_to_float(src_bits[idx]) * scalar);
    }
}

void
npu_golden_vpu_unary_i2f(const int32_t *src, uint32_t *dst_bits,
                         uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst_bits[idx] = npu_float_to_bits((float)src[idx]);
    }
}

void
npu_golden_vpu_unary_f2i(const uint32_t *src_bits, int32_t *dst,
                         uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = (int32_t)npu_bits_to_float(src_bits[idx]);
    }
}

void
npu_golden_vpu_unary_sqrt(const uint32_t *src_bits, uint32_t *dst_bits,
                          uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst_bits[idx] = npu_float_to_bits(
            sqrtf(npu_bits_to_float(src_bits[idx])));
    }
}

void
npu_golden_vpu_unary_exp(const uint32_t *src_bits, uint32_t *dst_bits,
                         uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst_bits[idx] = npu_float_to_bits(
            expf(npu_bits_to_float(src_bits[idx])));
    }
}
