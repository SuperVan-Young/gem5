/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "npu_sync.hh"
#include "vpu.hh"

enum SystemVpuDualLayout
{
    SYSTEM_VPU_DUAL_SLOT_STRIDE_BYTES = 0x40U,
    SYSTEM_VPU_DUAL_VPU0_ID = 0x0U,
    SYSTEM_VPU_DUAL_VPU1_ID = 0x1U,
    SYSTEM_VPU_DUAL_RELEASE_SYNC = 0x31U,
    SYSTEM_VPU_DUAL_VPU0_SYNC = 0x32U,
    SYSTEM_VPU_DUAL_VPU1_SYNC = 0x33U,
    SYSTEM_VPU_DUAL_ELEM_COUNT = 4U,
    SYSTEM_VPU_DUAL_VPU0_SRC = 0U,
    SYSTEM_VPU_DUAL_VPU0_DST = 1U,
    SYSTEM_VPU_DUAL_VPU1_SRC = 2U,
    SYSTEM_VPU_DUAL_VPU1_DST = 3U,
};

static volatile uint32_t *
spm_slot_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        0x60000000UL +
        ((uint64_t)port_id * SYSTEM_VPU_DUAL_SLOT_STRIDE_BYTES));
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

static uint32_t
exp_expected_bits(uint32_t idx)
{
    static const uint32_t expected[4] = {
        0x3ebc5ab2U, // exp(-1.0)
        0x3f1b4598U, // exp(-0.5)
        0x3f800000U, // exp(0.0)
        0x402df854U, // exp(1.0)
    };
    return expected[idx];
}

static float
float_abs(float value)
{
    return value < 0.0f ? -value : value;
}

static void
clear_slot(uint32_t port_id)
{
    volatile uint32_t *base = spm_slot_ptr(port_id);

    for (uint32_t idx = 0U;
         idx < SYSTEM_VPU_DUAL_SLOT_STRIDE_BYTES / sizeof(uint32_t); ++idx) {
        base[idx] = 0U;
    }
}

static void
store_u32_vector(uint32_t port_id, const uint32_t *values, uint32_t count)
{
    volatile uint32_t *base = spm_slot_ptr(port_id);

    for (uint32_t idx = 0U; idx < count; ++idx) {
        base[idx] = values[idx];
    }
}

static void
load_u32_vector(uint32_t port_id, uint32_t *values, uint32_t count)
{
    volatile uint32_t *base = spm_slot_ptr(port_id);

    for (uint32_t idx = 0U; idx < count; ++idx) {
        values[idx] = base[idx];
    }
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
            const float actual_value = bits_to_float(spm_slot_ptr(port_id)[idx]);
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
            if (spm_slot_ptr(port_id)[idx] != expected[idx]) {
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
verify_outputs_hold_zero(uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        for (uint32_t idx = 0U; idx < SYSTEM_VPU_DUAL_ELEM_COUNT; ++idx) {
            if (spm_slot_ptr(SYSTEM_VPU_DUAL_VPU0_DST)[idx] != 0U ||
                spm_slot_ptr(SYSTEM_VPU_DUAL_VPU1_DST)[idx] != 0U) {
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
    for (uint32_t idx = 0U; idx < SYSTEM_VPU_DUAL_ELEM_COUNT; ++idx) {
        printf(" [%u]=%#x", idx, values[idx]);
    }
    printf("\n");
}

int
main(void)
{
    const uint32_t vpu0_src[SYSTEM_VPU_DUAL_ELEM_COUNT] = {
        float_to_bits(1.0f),
        float_to_bits(-2.0f),
        float_to_bits(3.0f),
        float_to_bits(-4.0f),
    };
    const uint32_t vpu1_src[SYSTEM_VPU_DUAL_ELEM_COUNT] = {
        float_to_bits(-1.0f),
        float_to_bits(-0.5f),
        float_to_bits(0.0f),
        float_to_bits(1.0f),
    };
    uint32_t vpu0_expected[SYSTEM_VPU_DUAL_ELEM_COUNT];
    uint32_t vpu1_expected[SYSTEM_VPU_DUAL_ELEM_COUNT];
    uint32_t actual[SYSTEM_VPU_DUAL_ELEM_COUNT];

    clear_slot(SYSTEM_VPU_DUAL_VPU0_SRC);
    clear_slot(SYSTEM_VPU_DUAL_VPU0_DST);
    clear_slot(SYSTEM_VPU_DUAL_VPU1_SRC);
    clear_slot(SYSTEM_VPU_DUAL_VPU1_DST);
    store_u32_vector(SYSTEM_VPU_DUAL_VPU0_SRC, vpu0_src,
                     SYSTEM_VPU_DUAL_ELEM_COUNT);
    store_u32_vector(SYSTEM_VPU_DUAL_VPU1_SRC, vpu1_src,
                     SYSTEM_VPU_DUAL_ELEM_COUNT);

    for (uint32_t idx = 0U; idx < SYSTEM_VPU_DUAL_ELEM_COUNT; ++idx) {
        vpu0_expected[idx] = float_to_bits(bits_to_float(vpu0_src[idx]) * 2.0f);
        vpu1_expected[idx] = exp_expected_bits(idx);
    }

    npu_launch_sync_wait(SYSTEM_VPU_DUAL_VPU0_ID,
                         SYSTEM_VPU_DUAL_RELEASE_SYNC,
                         0x11111111U, 0x22222222U, 0x33333333U);
    vpu_cmd_launch_scale(SYSTEM_VPU_DUAL_VPU0_ID, SYSTEM_VPU_DUAL_VPU0_SYNC,
                         0x1U, 0x2U, 1U, SYSTEM_VPU_DUAL_ELEM_COUNT,
                         sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32,
                         float_to_bits(2.0f));
    vpu_cmd_launch_unary(SYSTEM_VPU_DUAL_VPU1_ID, VPU_OP_VEXP,
                         SYSTEM_VPU_DUAL_VPU1_SYNC, 0x4U, 0x8U, 1U,
                         SYSTEM_VPU_DUAL_ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (verify_outputs_hold_zero(4096ULL) != 0) {
        printf("SYSTEM_VPU_DUAL_EARLY_DISPATCH\n");
        printf("SYSTEM_VPU_DUAL_TEST_FAIL\n");
        return 1;
    }

    npu_sync_signal_set(SYSTEM_VPU_DUAL_VPU0_ID,
                        SYSTEM_VPU_DUAL_RELEASE_SYNC);
    npu_cmd_sync_done();

    if (wait_vector_match(SYSTEM_VPU_DUAL_VPU0_DST, vpu0_expected,
                          SYSTEM_VPU_DUAL_ELEM_COUNT, 80000000ULL) != 0) {
        load_u32_vector(SYSTEM_VPU_DUAL_VPU0_DST, actual,
                        SYSTEM_VPU_DUAL_ELEM_COUNT);
        print_vector("SYSTEM_VPU_DUAL_VPU0_EXPECTED", vpu0_expected);
        print_vector("SYSTEM_VPU_DUAL_VPU0_ACTUAL", actual);
        printf("SYSTEM_VPU_DUAL_TEST_FAIL\n");
        return 1;
    }

    if (wait_float_vector_close(SYSTEM_VPU_DUAL_VPU1_DST, vpu1_expected,
                                SYSTEM_VPU_DUAL_ELEM_COUNT, 80000000ULL,
                                0.03f, 0.03f) != 0) {
        load_u32_vector(SYSTEM_VPU_DUAL_VPU1_DST, actual,
                        SYSTEM_VPU_DUAL_ELEM_COUNT);
        print_vector("SYSTEM_VPU_DUAL_VPU1_EXPECTED", vpu1_expected);
        print_vector("SYSTEM_VPU_DUAL_VPU1_ACTUAL", actual);
        printf("SYSTEM_VPU_DUAL_TEST_FAIL\n");
        return 1;
    }

    print_vector("SYSTEM_VPU_DUAL_VPU0_FINAL", vpu0_expected);
    print_vector("SYSTEM_VPU_DUAL_VPU1_FINAL", vpu1_expected);
    printf("SYSTEM_VPU_DUAL_VPU0_EXPECTED_COMPLETED=1\n");
    printf("SYSTEM_VPU_DUAL_VPU0_EXPECTED_PROLOGUES=1\n");
    printf("SYSTEM_VPU_DUAL_VPU0_EXPECTED_EXECUTES=1\n");
    printf("SYSTEM_VPU_DUAL_VPU0_EXPECTED_EPILOGUES=1\n");
    printf("SYSTEM_VPU_DUAL_VPU0_EXPECTED_ITERATIONS=1\n");
    printf("SYSTEM_VPU_DUAL_VPU0_EXPECTED_READ_RESPS=1\n");
    printf("SYSTEM_VPU_DUAL_VPU0_EXPECTED_WRITE_RESPS=1\n");
    printf("SYSTEM_VPU_DUAL_VPU1_EXPECTED_COMPLETED=1\n");
    printf("SYSTEM_VPU_DUAL_VPU1_EXPECTED_PROLOGUES=1\n");
    printf("SYSTEM_VPU_DUAL_VPU1_EXPECTED_EXECUTES=1\n");
    printf("SYSTEM_VPU_DUAL_VPU1_EXPECTED_EPILOGUES=1\n");
    printf("SYSTEM_VPU_DUAL_VPU1_EXPECTED_ITERATIONS=1\n");
    printf("SYSTEM_VPU_DUAL_VPU1_EXPECTED_READ_RESPS=1\n");
    printf("SYSTEM_VPU_DUAL_VPU1_EXPECTED_WRITE_RESPS=1\n");
    printf("SYSTEM_VPU_DUAL_TEST_PASS\n");
    return 0;
}
