/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "golden/vpu_unary.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "vpu.hh"

enum SystemVpuDualLayout
{
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

static int
verify_outputs_hold_zero(uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        for (uint32_t idx = 0U; idx < SYSTEM_VPU_DUAL_ELEM_COUNT; ++idx) {
            if (npu_spm_slot_word_ptr_default(SYSTEM_VPU_DUAL_VPU0_DST)[idx] !=
                    0U ||
                npu_spm_slot_word_ptr_default(SYSTEM_VPU_DUAL_VPU1_DST)[idx] !=
                    0U) {
                return -1;
            }
        }
    }

    return 0;
}

static void
print_vector(const char *prefix, const uint32_t *values)
{
    npu_dump_u32_vector(prefix, values, SYSTEM_VPU_DUAL_ELEM_COUNT);
}

int
main(void)
{
    const uint32_t vpu0_src[SYSTEM_VPU_DUAL_ELEM_COUNT] = {
        npu_float_to_bits(1.0f),
        npu_float_to_bits(-2.0f),
        npu_float_to_bits(3.0f),
        npu_float_to_bits(-4.0f),
    };
    const uint32_t vpu1_src[SYSTEM_VPU_DUAL_ELEM_COUNT] = {
        npu_float_to_bits(-1.0f),
        npu_float_to_bits(-0.5f),
        npu_float_to_bits(0.0f),
        npu_float_to_bits(1.0f),
    };
    uint32_t vpu0_expected[SYSTEM_VPU_DUAL_ELEM_COUNT];
    uint32_t vpu1_expected[SYSTEM_VPU_DUAL_ELEM_COUNT];
    uint32_t actual[SYSTEM_VPU_DUAL_ELEM_COUNT];

    npu_spm_clear_slot(SYSTEM_VPU_DUAL_VPU0_SRC);
    npu_spm_clear_slot(SYSTEM_VPU_DUAL_VPU0_DST);
    npu_spm_clear_slot(SYSTEM_VPU_DUAL_VPU1_SRC);
    npu_spm_clear_slot(SYSTEM_VPU_DUAL_VPU1_DST);
    npu_spm_store_u32_vector(SYSTEM_VPU_DUAL_VPU0_SRC, vpu0_src,
                             SYSTEM_VPU_DUAL_ELEM_COUNT);
    npu_spm_store_u32_vector(SYSTEM_VPU_DUAL_VPU1_SRC, vpu1_src,
                             SYSTEM_VPU_DUAL_ELEM_COUNT);

    npu_golden_vpu_unary_scale_f32(vpu0_src, vpu0_expected,
                                   SYSTEM_VPU_DUAL_ELEM_COUNT, 1U,
                                   npu_float_to_bits(2.0f));
    npu_golden_vpu_unary_exp(vpu1_src, vpu1_expected,
                             SYSTEM_VPU_DUAL_ELEM_COUNT);

    npu_launch_sync_wait(SYSTEM_VPU_DUAL_VPU0_ID,
                         SYSTEM_VPU_DUAL_RELEASE_SYNC, 0x11111111U,
                         0x22222222U, 0x33333333U);
    vpu_cmd_launch_scale(SYSTEM_VPU_DUAL_VPU0_ID, SYSTEM_VPU_DUAL_VPU0_SYNC,
                         0x1U, 0x2U, 1U, SYSTEM_VPU_DUAL_ELEM_COUNT,
                         sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32,
                         npu_float_to_bits(2.0f));
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

    if (npu_wait_u32_vector_match(NULL,
                                  npu_spm_slot_word_ptr_default(
                                      SYSTEM_VPU_DUAL_VPU0_DST),
                                  vpu0_expected, SYSTEM_VPU_DUAL_ELEM_COUNT,
                                  80000000ULL) != 0) {
        npu_spm_load_u32_vector(SYSTEM_VPU_DUAL_VPU0_DST, actual,
                                SYSTEM_VPU_DUAL_ELEM_COUNT);
        print_vector("SYSTEM_VPU_DUAL_VPU0_EXPECTED", vpu0_expected);
        print_vector("SYSTEM_VPU_DUAL_VPU0_ACTUAL", actual);
        printf("SYSTEM_VPU_DUAL_TEST_FAIL\n");
        return 1;
    }

    if (npu_wait_float_vector_close(NULL,
                                    npu_spm_slot_word_ptr_default(
                                        SYSTEM_VPU_DUAL_VPU1_DST),
                                    vpu1_expected, SYSTEM_VPU_DUAL_ELEM_COUNT,
                                    80000000ULL, 0.03f, 0.03f) != 0) {
        npu_spm_load_u32_vector(SYSTEM_VPU_DUAL_VPU1_DST, actual,
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
