#ifndef TESTS_GEM5_NPU_UTILS_NPU_ASSERT_H_
#define TESTS_GEM5_NPU_UTILS_NPU_ASSERT_H_

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static inline uint32_t
npu_float_to_bits(float value)
{
    uint32_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline float
npu_bits_to_float(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline float
npu_float_abs(float value)
{
    return value < 0.0f ? -value : value;
}

static inline int
npu_float_close(float expected, float actual, float abs_tol, float rel_tol)
{
    const float diff = fabsf(expected - actual);
    const float limit = abs_tol + (rel_tol * fabsf(expected));
    return diff <= limit;
}

static inline void
npu_dump_u32_vector(const char *prefix, const uint32_t *values,
                    uint32_t count)
{
    printf("%s", prefix);
    for (uint32_t idx = 0U; idx < count; ++idx) {
        printf(" [%u]=%#x", idx, values[idx]);
    }
    printf("\n");
}

static inline void
npu_dump_u32_vector_from_volatile(const char *prefix,
                                  const volatile uint32_t *values,
                                  uint32_t count)
{
    printf("%s", prefix);
    for (uint32_t idx = 0U; idx < count; ++idx) {
        printf(" [%u]=%#x", idx, values[idx]);
    }
    printf("\n");
}

static inline void
npu_dump_u32_scalar(const char *label, uint32_t value)
{
    printf("%s=%#x\n", label, value);
}

static inline int
npu_expect_u32_equal(const char *label, uint32_t expected, uint32_t actual)
{
    if (expected == actual) {
        return 0;
    }

    printf("%s expected=%#x actual=%#x\n", label, expected, actual);
    return -1;
}

static inline int
npu_expect_u32_vector(const char *label, const uint32_t *expected,
                      const uint32_t *actual, uint32_t count)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        if (expected[idx] != actual[idx]) {
            printf("%s mismatch\n", label);
            npu_dump_u32_vector("  expected", expected, count);
            npu_dump_u32_vector("  actual", actual, count);
            return -1;
        }
    }

    return 0;
}

static inline int
npu_expect_float_vector_close(const char *label, const uint32_t *expected,
                              const uint32_t *actual, uint32_t count,
                              float abs_tol, float rel_tol)
{
    for (uint32_t idx = 0U; idx < count; ++idx) {
        const float expected_value = npu_bits_to_float(expected[idx]);
        const float actual_value = npu_bits_to_float(actual[idx]);
        if (!npu_float_close(expected_value, actual_value, abs_tol, rel_tol)) {
            printf("%s mismatch at [%u]\n", label, idx);
            npu_dump_u32_vector("  expected", expected, count);
            npu_dump_u32_vector("  actual", actual, count);
            return -1;
        }
    }

    return 0;
}

static inline int
npu_wait_u32_scalar_match(const char *label, const volatile uint32_t *actual,
                          uint32_t expected, uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        if (*actual == expected) {
            return 0;
        }
    }

    if (label != NULL) {
        printf("%s timeout expected=%#x actual=%#x\n", label, expected,
               *actual);
    }
    return -1;
}

static inline int
npu_wait_u32_vector_match(const char *label, const volatile uint32_t *actual,
                          const uint32_t *expected, uint32_t count,
                          uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            if (actual[idx] != expected[idx]) {
                matched = 0;
                break;
            }
        }

        if (matched) {
            return 0;
        }
    }

    if (label != NULL) {
        printf("%s timeout\n", label);
        npu_dump_u32_vector("  expected", expected, count);
        npu_dump_u32_vector_from_volatile("  actual", actual, count);
    }
    return -1;
}

static inline int
npu_wait_i32_vector_not_value(const char *label,
                              const volatile int32_t *actual,
                              int32_t forbidden, uint32_t count,
                              uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int ready = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            if (actual[idx] == forbidden) {
                ready = 0;
                break;
            }
        }

        if (ready) {
            return 0;
        }
    }

    if (label != NULL) {
        printf("%s timeout forbidden=%d\n", label, forbidden);
        for (uint32_t idx = 0U; idx < count; ++idx) {
            printf("  actual[%u]=%d\n", idx, actual[idx]);
        }
    }
    return -1;
}

static inline int
npu_wait_float_vector_close(const char *label,
                            const volatile uint32_t *actual,
                            const uint32_t *expected, uint32_t count,
                            uint64_t timeout, float abs_tol, float rel_tol)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            const float expected_value = npu_bits_to_float(expected[idx]);
            const float actual_value = npu_bits_to_float(actual[idx]);
            if (!npu_float_close(expected_value, actual_value, abs_tol,
                                 rel_tol)) {
                matched = 0;
                break;
            }
        }

        if (matched) {
            return 0;
        }
    }

    if (label != NULL) {
        printf("%s timeout\n", label);
        npu_dump_u32_vector("  expected", expected, count);
        npu_dump_u32_vector_from_volatile("  actual", actual, count);
    }
    return -1;
}

#endif
