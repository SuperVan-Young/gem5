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

static inline uint32_t
llm_slot_span_bytes(uint64_t bytes)
{
    return static_cast<uint32_t>(
        (bytes + VPU_LOCAL_SLOT_STRIDE - 1U) / VPU_LOCAL_SLOT_STRIDE);
}

static inline uint32_t
llm_matrix_slot_span(uint32_t rows, uint32_t cols)
{
    return llm_slot_span_bytes(
        static_cast<uint64_t>(rows) * cols * sizeof(uint32_t));
}

static inline uint32_t
llm_vector_slot_span(uint32_t elems)
{
    return llm_slot_span_bytes(static_cast<uint64_t>(elems) * sizeof(uint32_t));
}

static inline void
llm_clear_slot_span(uint32_t base_slot, uint32_t slot_span)
{
    for (uint32_t slot = 0U; slot < slot_span; ++slot) {
        npu_spm_clear_slot(base_slot + slot);
    }
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
