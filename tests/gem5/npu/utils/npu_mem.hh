#ifndef TESTS_GEM5_NPU_UTILS_NPU_MEM_H_
#define TESTS_GEM5_NPU_UTILS_NPU_MEM_H_

#include <stdint.h>

#define NPU_MEM_DEFAULT_SPM_BASE_ADDR 0x60000000UL
#define NPU_MEM_DEFAULT_SPM_SLOT_STRIDE_BYTES 0x40U

static inline volatile uint32_t *
npu_mem_word_ptr(uint64_t base_addr, uint32_t byte_offset)
{
    return (volatile uint32_t *)(uintptr_t)(base_addr + byte_offset);
}

static inline volatile uint32_t *
npu_spm_slot_word_ptr(uint32_t port_id, uint64_t base_addr,
                      uint32_t slot_stride_bytes)
{
    return npu_mem_word_ptr(base_addr, port_id * slot_stride_bytes);
}

static inline volatile uint32_t *
npu_spm_slot_word_ptr_default(uint32_t port_id)
{
    return npu_spm_slot_word_ptr(port_id, NPU_MEM_DEFAULT_SPM_BASE_ADDR,
                                 NPU_MEM_DEFAULT_SPM_SLOT_STRIDE_BYTES);
}

static inline void
npu_mem_clear_u32(volatile uint32_t *dst, uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = 0U;
    }
}

static inline void
npu_mem_fill_u32(volatile uint32_t *dst, uint32_t value, uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = value;
    }
}

static inline void
npu_mem_store_u32(volatile uint32_t *dst, const uint32_t *src, uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = src[idx];
    }
}

static inline void
npu_mem_load_u32(const volatile uint32_t *src, uint32_t *dst, uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = src[idx];
    }
}

static inline void
npu_mem_clear_bytes(volatile uint8_t *dst, uint32_t byte_count)
{
    for (uint32_t idx = 0U; idx < byte_count; ++idx) {
        dst[idx] = 0U;
    }
}

static inline void
npu_spm_clear_slot(uint32_t port_id)
{
    npu_mem_clear_u32(npu_spm_slot_word_ptr_default(port_id),
                      NPU_MEM_DEFAULT_SPM_SLOT_STRIDE_BYTES /
                          sizeof(uint32_t));
}

static inline void
npu_spm_fill_u32_slot(uint32_t port_id, uint32_t value, uint32_t count)
{
    npu_mem_fill_u32(npu_spm_slot_word_ptr_default(port_id), value, count);
}

static inline void
npu_spm_store_u32_vector(uint32_t port_id, const uint32_t *src, uint32_t count)
{
    npu_mem_store_u32(npu_spm_slot_word_ptr_default(port_id), src, count);
}

static inline void
npu_spm_load_u32_vector(uint32_t port_id, uint32_t *dst, uint32_t count)
{
    npu_mem_load_u32(npu_spm_slot_word_ptr_default(port_id), dst, count);
}

#endif
