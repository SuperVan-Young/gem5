#ifndef TESTS_GEM5_NPU_UTILS_NPU_MMIO_H_
#define TESTS_GEM5_NPU_UTILS_NPU_MMIO_H_

#include <stdint.h>

static inline void
npu_mmio_write32(uint64_t addr, const uint32_t *data, unsigned count)
{
    volatile uint32_t *mmio = (volatile uint32_t *)addr;

    for (unsigned i = 0; i < count; ++i) {
        mmio[i] = data[i];
    }
}

static inline void
npu_mmio_read32(uint64_t addr, uint32_t *data, unsigned count)
{
    volatile const uint32_t *mmio = (volatile const uint32_t *)addr;

    for (unsigned i = 0; i < count; ++i) {
        data[i] = mmio[i];
    }
}

static inline void
npu_mmio_write32_one(uint64_t addr, uint32_t value)
{
    npu_mmio_write32(addr, &value, 1U);
}

static inline uint32_t
npu_mmio_read32_one(uint64_t addr)
{
    uint32_t value = 0;

    npu_mmio_read32(addr, &value, 1U);
    return value;
}

#endif
