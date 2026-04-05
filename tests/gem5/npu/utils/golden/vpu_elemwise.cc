#include "vpu_elemwise.hh"

#include <limits.h>
#include <stdint.h>

#include "../cmd/vpu.hh"
#include "../npu_assert.hh"

static int32_t
npu_golden_vpu_elemwise_div_i32(int32_t lhs, int32_t rhs)
{
    if (rhs == 0) {
        return 0;
    }

    if (lhs == INT32_MIN && rhs == -1) {
        return INT32_MAX;
    }

    return lhs / rhs;
}

void
npu_golden_vpu_elemwise_i32(uint32_t op_code, const uint32_t *lhs,
                            const uint32_t *rhs, uint32_t *dst,
                            uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        switch (op_code) {
          case VPU_OP_VADD:
            dst[idx] = lhs[idx] + rhs[idx];
            break;
          case VPU_OP_VSUB:
            dst[idx] = lhs[idx] - rhs[idx];
            break;
          case VPU_OP_VMUL:
            dst[idx] = lhs[idx] * rhs[idx];
            break;
          case VPU_OP_VDIV: {
            const int32_t value = npu_golden_vpu_elemwise_div_i32(
                (int32_t)lhs[idx], (int32_t)rhs[idx]);
            dst[idx] = (uint32_t)value;
            break;
          }
          default:
            dst[idx] = 0U;
            break;
        }
    }
}

void
npu_golden_vpu_elemwise_f32(uint32_t op_code, const uint32_t *lhs_bits,
                            const uint32_t *rhs_bits, uint32_t *dst_bits,
                            uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        const float lhs = npu_bits_to_float(lhs_bits[idx]);
        const float rhs = npu_bits_to_float(rhs_bits[idx]);
        float value = 0.0f;

        switch (op_code) {
          case VPU_OP_VADD:
            value = lhs + rhs;
            break;
          case VPU_OP_VSUB:
            value = lhs - rhs;
            break;
          case VPU_OP_VMUL:
            value = lhs * rhs;
            break;
          case VPU_OP_VDIV:
            value = lhs / rhs;
            break;
          default:
            value = 0.0f;
            break;
        }

        dst_bits[idx] = npu_float_to_bits(value);
    }
}
