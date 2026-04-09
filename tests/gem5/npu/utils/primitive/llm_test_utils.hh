#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_LLM_TEST_UTILS_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_LLM_TEST_UTILS_HH_

#include <stdint.h>

#include "../npu_mem.hh"
#include "../tensor.hh"

static inline PrimitiveTensorDesc
llm_packed_last_axis_tensor(uint32_t slot, uint32_t rows, uint32_t cols)
{
    return PrimitiveTensorDesc::denseSpm(slot, VPU_DATA_F32, {cols, rows})
        .permute({1U, 0U});
}

static inline void
llm_store_logical_matrix_last_axis_front(uint32_t slot, const uint32_t *src_bits,
                                         uint32_t rows, uint32_t cols)
{
    volatile uint32_t *dst = npu_spm_slot_word_ptr_default(slot);
    for (uint32_t row = 0U; row < rows; ++row) {
        for (uint32_t col = 0U; col < cols; ++col) {
            dst[col * rows + row] = src_bits[row * cols + col];
        }
    }
}

static inline void
llm_load_logical_matrix_last_axis_front(uint32_t slot, uint32_t *dst_bits,
                                        uint32_t rows, uint32_t cols)
{
    const volatile uint32_t *src = npu_spm_slot_word_ptr_default(slot);
    for (uint32_t row = 0U; row < rows; ++row) {
        for (uint32_t col = 0U; col < cols; ++col) {
            dst_bits[row * cols + col] = src[col * rows + row];
        }
    }
}

#endif
