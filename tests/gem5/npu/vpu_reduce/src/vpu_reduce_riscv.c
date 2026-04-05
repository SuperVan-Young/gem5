/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "golden/vpu_reduce.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum ReduceLayout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    INT_SUM_SYNC = 0x71U,
    FP_SUM_SYNC = 0x72U,
    FP_MAX_SYNC = 0x73U,
};

struct VpuStatsExpectation
{
    uint32_t completed_cmds;
    uint32_t prologues;
    uint32_t executes;
    uint32_t epilogues;
    uint32_t read_resps;
    uint32_t write_resps;
    uint32_t iterations;
};

static void
accumulate_expected_stats(struct VpuStatsExpectation *stats,
                          uint32_t repetition)
{
    stats->completed_cmds += 1U;
    stats->prologues += repetition;
    stats->executes += repetition;
    stats->epilogues += repetition;
    stats->read_resps += repetition;
    stats->write_resps += repetition;
    stats->iterations += repetition;
}

static int
run_int_reduce_sum_case(void)
{
    const int32_t src_i32[ELEM_COUNT] = {2, -3, 4, 7};
    uint32_t src[ELEM_COUNT];
    int32_t expected_i32 = 0;
    uint32_t expected = 0;

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src[idx], &src_i32[idx], sizeof(src[idx]));
    }
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_reduce_sum_i32(src_i32, &expected_i32, ELEM_COUNT);
    memcpy(&expected, &expected_i32, sizeof(expected));

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VREDUCE_SUM, INT_SUM_SYNC,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_I32);

    if (npu_wait_u32_scalar_match(NULL, npu_spm_slot_word_ptr_default(DST_PORT),
                                  expected, 60000000ULL) != 0) {
        printf("VPU_REDUCE_FAIL int_sum exp=%d act=%d\n", expected_i32,
               (int32_t)npu_spm_slot_word_ptr_default(DST_PORT)[0]);
        return -1;
    }

    return 0;
}

static int
run_fp_reduce_sum_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        npu_float_to_bits(1.5f),
        npu_float_to_bits(-2.0f),
        npu_float_to_bits(4.0f),
        npu_float_to_bits(0.25f),
    };
    uint32_t expected = 0;

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_reduce_sum_f32(src, &expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VREDUCE_SUM, FP_SUM_SYNC,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_u32_scalar_match(NULL, npu_spm_slot_word_ptr_default(DST_PORT),
                                  expected, 60000000ULL) != 0) {
        printf("VPU_REDUCE_FAIL fp_sum exp=%#x act=%#x\n", expected,
               npu_spm_slot_word_ptr_default(DST_PORT)[0]);
        return -1;
    }

    return 0;
}

static int
run_fp_reduce_max_case(void)
{
    const uint32_t src[ELEM_COUNT] = {
        npu_float_to_bits(-3.0f),
        npu_float_to_bits(5.5f),
        npu_float_to_bits(2.25f),
        npu_float_to_bits(4.0f),
    };
    uint32_t expected = 0;

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_reduce_max_f32(src, &expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VREDUCE_MAX, FP_MAX_SYNC,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_u32_scalar_match(NULL, npu_spm_slot_word_ptr_default(DST_PORT),
                                  expected, 60000000ULL) != 0) {
        printf("VPU_REDUCE_FAIL fp_max exp=%#x act=%#x\n", expected,
               npu_spm_slot_word_ptr_default(DST_PORT)[0]);
        return -1;
    }

    return 0;
}

int
main(void)
{
    struct VpuStatsExpectation stats = {0};

    if (run_int_reduce_sum_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_fp_reduce_sum_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_fp_reduce_max_case() != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    printf("VPU_REDUCE_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_REDUCE_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_REDUCE_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_REDUCE_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_REDUCE_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_REDUCE_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_REDUCE_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_REDUCE_PASS\n");
    return 0;
}
