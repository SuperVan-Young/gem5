#include "mpu_gemm.hh"

void
npu_golden_mpu_gemm_i8_i8_i32(const int8_t *a_data, const int8_t *b_data,
                              int32_t *c_data, uint32_t m, uint32_t n,
                              uint32_t k, uint32_t a_row_stride_elems,
                              uint32_t b_row_stride_elems,
                              uint32_t c_row_stride_elems)
{
    for (uint32_t row = 0U; row < m; ++row) {
        for (uint32_t col = 0U; col < n; ++col) {
            int32_t acc = 0;
            for (uint32_t depth = 0U; depth < k; ++depth) {
                const int32_t a_value =
                    (int32_t)a_data[row * a_row_stride_elems + depth];
                const int32_t b_value =
                    (int32_t)b_data[depth * b_row_stride_elems + col];
                acc += a_value * b_value;
            }
            c_data[row * c_row_stride_elems + col] = acc;
        }
    }
}
