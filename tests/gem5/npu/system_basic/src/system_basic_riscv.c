/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "npu_sync.hh"
#include "vpu.hh"

enum SystemBasicLayout
{
    SYSTEM_BASIC_NUM_PORTS = 3U,
    SYSTEM_BASIC_SLOT_STRIDE_BYTES = 0x40U,
    SYSTEM_BASIC_DEVICE_ID = 0x0U,
    SYSTEM_BASIC_RELEASE_SYNC = 0x2AU,
    SYSTEM_BASIC_LINEAR_SYNC = 0x2BU,
    SYSTEM_BASIC_LUT_SYNC = 0x2CU,
    SYSTEM_BASIC_SRC_PORT = 0U,
    SYSTEM_BASIC_LINEAR_DST_PORT = 1U,
    SYSTEM_BASIC_LUT_DST_PORT = 2U,
    SYSTEM_BASIC_ELEM_COUNT = 4U,
};

struct ExpectedState
{
    uint32_t src[SYSTEM_BASIC_ELEM_COUNT];
    uint32_t linear[SYSTEM_BASIC_ELEM_COUNT];
    uint32_t lut[SYSTEM_BASIC_ELEM_COUNT];
};

static volatile uint32_t *
slot_word_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        0x60000000UL +
        ((uint64_t)port_id * SYSTEM_BASIC_SLOT_STRIDE_BYTES));
}

static uint32_t
float_to_bits(float value)
{
    uint32_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float
bits_to_float(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void
clear_slot(uint32_t port_id)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U;
         idx < SYSTEM_BASIC_SLOT_STRIDE_BYTES / sizeof(uint32_t); ++idx) {
        base[idx] = 0U;
    }
}

static void
store_u32_vector(uint32_t port_id, const uint32_t *values, uint32_t count)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U; idx < count; ++idx) {
        base[idx] = values[idx];
    }
}

static void
load_u32_vector(uint32_t port_id, uint32_t *values, uint32_t count)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U; idx < count; ++idx) {
        values[idx] = base[idx];
    }
}

static float
float_abs(float value)
{
    return value < 0.0f ? -value : value;
}

static int
float_close(float expected, float actual, float abs_tol, float rel_tol)
{
    const float diff = float_abs(expected - actual);
    const float limit = abs_tol + (rel_tol * float_abs(expected));
    return diff <= limit;
}

static int
wait_float_vector_close(uint32_t port_id, const uint32_t *expected,
                        uint32_t count, uint64_t timeout, float abs_tol,
                        float rel_tol)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            const float expected_value = bits_to_float(expected[idx]);
            const float actual_value = bits_to_float(slot_word_ptr(port_id)[idx]);
            if (!float_close(expected_value, actual_value, abs_tol, rel_tol)) {
                matched = 0;
                break;
            }
        }

        if (matched) {
            return 0;
        }
    }

    return -1;
}

static int
wait_vector_match(uint32_t port_id, const uint32_t *expected, uint32_t count,
                  uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            if (slot_word_ptr(port_id)[idx] != expected[idx]) {
                matched = 0;
                break;
            }
        }

        if (matched) {
            return 0;
        }
    }

    return -1;
}

static int
verify_state_unchanged_before_release(const struct ExpectedState *expected,
                                      uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        for (uint32_t idx = 0U; idx < SYSTEM_BASIC_ELEM_COUNT; ++idx) {
            if (slot_word_ptr(SYSTEM_BASIC_SRC_PORT)[idx] != expected->src[idx] ||
                slot_word_ptr(SYSTEM_BASIC_LINEAR_DST_PORT)[idx] != 0U ||
                slot_word_ptr(SYSTEM_BASIC_LUT_DST_PORT)[idx] != 0U) {
                return -1;
            }
        }
    }

    return 0;
}

static void
print_vector(const char *prefix, const uint32_t *values)
{
    printf("%s", prefix);
    for (uint32_t idx = 0U; idx < SYSTEM_BASIC_ELEM_COUNT; ++idx) {
        printf(" [%u]=%#x", idx, values[idx]);
    }
    printf("\n");
}

int
main(void)
{
    const float scale = 4.0f;
    const uint32_t src[SYSTEM_BASIC_ELEM_COUNT] = {
        float_to_bits(0.25f),
        float_to_bits(1.0f),
        float_to_bits(4.0f),
        float_to_bits(9.0f),
    };
    struct ExpectedState expected = {0};
    uint32_t actual[SYSTEM_BASIC_ELEM_COUNT];

    clear_slot(SYSTEM_BASIC_SRC_PORT);
    clear_slot(SYSTEM_BASIC_LINEAR_DST_PORT);
    clear_slot(SYSTEM_BASIC_LUT_DST_PORT);
    memcpy(expected.src, src, sizeof(src));
    store_u32_vector(SYSTEM_BASIC_SRC_PORT, src, SYSTEM_BASIC_ELEM_COUNT);

    for (uint32_t idx = 0U; idx < SYSTEM_BASIC_ELEM_COUNT; ++idx) {
        const float linear = bits_to_float(src[idx]) * scale;
        expected.linear[idx] = float_to_bits(linear);
        switch (idx) {
          case 0:
            expected.lut[idx] = float_to_bits(1.0f);
            break;
          case 1:
            expected.lut[idx] = float_to_bits(2.0f);
            break;
          case 2:
            expected.lut[idx] = float_to_bits(4.0f);
            break;
          case 3:
            expected.lut[idx] = float_to_bits(6.0f);
            break;
          default:
            expected.lut[idx] = 0U;
            break;
        }
    }

    npu_launch_sync_wait(SYSTEM_BASIC_DEVICE_ID, SYSTEM_BASIC_RELEASE_SYNC,
                         0x11111111U, 0x22222222U, 0x33333333U);
    vpu_cmd_launch_scale(SYSTEM_BASIC_DEVICE_ID, SYSTEM_BASIC_LINEAR_SYNC,
                         0x1U, 0x2U, 1U, SYSTEM_BASIC_ELEM_COUNT,
                         sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32,
                         float_to_bits(scale));
    npu_launch_sync_wait(SYSTEM_BASIC_DEVICE_ID, SYSTEM_BASIC_LINEAR_SYNC,
                         0x44444444U, 0x55555555U, 0x66666666U);
    vpu_cmd_launch_unary(SYSTEM_BASIC_DEVICE_ID, VPU_OP_VSQRT,
                         SYSTEM_BASIC_LUT_SYNC, 0x2U, 0x4U, 1U,
                         SYSTEM_BASIC_ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (verify_state_unchanged_before_release(&expected, 4096ULL) != 0) {
        printf("SYSTEM_BASIC_EARLY_DISPATCH\n");
        printf("SYSTEM_BASIC_TEST_FAIL\n");
        return 1;
    }

    npu_sync_signal_set(SYSTEM_BASIC_DEVICE_ID, SYSTEM_BASIC_RELEASE_SYNC);
    npu_cmd_sync_done();

    if (wait_vector_match(SYSTEM_BASIC_LINEAR_DST_PORT, expected.linear,
                          SYSTEM_BASIC_ELEM_COUNT, 80000000ULL) != 0) {
        load_u32_vector(SYSTEM_BASIC_LINEAR_DST_PORT, actual,
                        SYSTEM_BASIC_ELEM_COUNT);
        print_vector("SYSTEM_BASIC_LINEAR_EXPECTED", expected.linear);
        print_vector("SYSTEM_BASIC_LINEAR_ACTUAL", actual);
        printf("SYSTEM_BASIC_TEST_FAIL\n");
        return 1;
    }

    if (wait_float_vector_close(SYSTEM_BASIC_LUT_DST_PORT, expected.lut,
                                SYSTEM_BASIC_ELEM_COUNT, 80000000ULL, 0.02f,
                                0.02f) != 0) {
        load_u32_vector(SYSTEM_BASIC_LUT_DST_PORT, actual,
                        SYSTEM_BASIC_ELEM_COUNT);
        print_vector("SYSTEM_BASIC_LUT_EXPECTED", expected.lut);
        print_vector("SYSTEM_BASIC_LUT_ACTUAL", actual);
        printf("SYSTEM_BASIC_TEST_FAIL\n");
        return 1;
    }

    print_vector("SYSTEM_BASIC_LINEAR_FINAL", expected.linear);
    print_vector("SYSTEM_BASIC_LUT_FINAL", expected.lut);
    printf("SYSTEM_BASIC_EXPECTED_COMPLETED_CMDS=2\n");
    printf("SYSTEM_BASIC_EXPECTED_PROLOGUES=2\n");
    printf("SYSTEM_BASIC_EXPECTED_EXECUTES=2\n");
    printf("SYSTEM_BASIC_EXPECTED_EPILOGUES=2\n");
    printf("SYSTEM_BASIC_EXPECTED_ITERATIONS=2\n");
    printf("SYSTEM_BASIC_EXPECTED_READ_RESPS=2\n");
    printf("SYSTEM_BASIC_EXPECTED_WRITE_RESPS=2\n");
    printf("SYSTEM_BASIC_TEST_PASS\n");
    return 0;
}
