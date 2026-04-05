/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <limits.h>
#include <stdio.h>

#include "golden/vpu_elemwise.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum ElemwiseLayout
{
    VPU_DEVICE_ID = 0U,
    SRC0_PORT = 0U,
    SRC1_PORT = 1U,
    DST_PORT = 2U,
    ELEM_COUNT = 4U,
    INT_ADD_SYNC = 0x41U,
    INT_SUB_SYNC = 0x42U,
    INT_MUL_SYNC = 0x43U,
    INT_DIV_SYNC = 0x44U,
    FP_ADD_SYNC = 0x45U,
    FP_SUB_SYNC = 0x46U,
    FP_MUL_SYNC = 0x47U,
    FP_DIV_SYNC = 0x48U,
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
    stats->read_resps += 2U * repetition;
    stats->write_resps += repetition;
    stats->iterations += repetition;
}

static int
run_int_case(uint32_t op_code, uint32_t sync_indicator,
             const uint32_t *lhs, const uint32_t *rhs, uint32_t *expected,
             uint32_t repetition, uint32_t dst_port)
{
    uint32_t actual[ELEM_COUNT];
    const uint32_t write_mask = 1U << dst_port;

    npu_spm_clear_slot(SRC0_PORT);
    npu_spm_clear_slot(SRC1_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC0_PORT, lhs, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC1_PORT, rhs, ELEM_COUNT);
    npu_golden_vpu_elemwise_i32(op_code, lhs, rhs, expected, ELEM_COUNT);

    vpu_cmd_launch_binary(VPU_DEVICE_ID, op_code, sync_indicator, 0x3U,
                          write_mask, repetition, ELEM_COUNT,
                          sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_I32);

    if (npu_wait_u32_vector_match(NULL,
                                  npu_spm_slot_word_ptr_default(dst_port),
                                  expected, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(dst_port, actual, ELEM_COUNT);
        printf("VPU_ELEMWISE_FAIL int op=%u\n", op_code);
        npu_expect_u32_vector("VPU_ELEMWISE_INT", expected, actual,
                              ELEM_COUNT);
        return -1;
    }

    return 0;
}

static int
run_float_case(uint32_t op_code, uint32_t sync_indicator,
               const uint32_t *lhs_bits, const uint32_t *rhs_bits,
               uint32_t *expected_bits, uint32_t repetition,
               uint32_t dst_port)
{
    uint32_t actual[ELEM_COUNT];
    const uint32_t write_mask = 1U << dst_port;

    npu_spm_clear_slot(SRC0_PORT);
    npu_spm_clear_slot(SRC1_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC0_PORT, lhs_bits, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC1_PORT, rhs_bits, ELEM_COUNT);
    npu_golden_vpu_elemwise_f32(op_code, lhs_bits, rhs_bits, expected_bits,
                                ELEM_COUNT);

    vpu_cmd_launch_binary(VPU_DEVICE_ID, op_code, sync_indicator, 0x3U,
                          write_mask, repetition, ELEM_COUNT,
                          sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_u32_vector_match(NULL,
                                  npu_spm_slot_word_ptr_default(dst_port),
                                  expected_bits, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(dst_port, actual, ELEM_COUNT);
        printf("VPU_ELEMWISE_FAIL float op=%u\n", op_code);
        npu_expect_u32_vector("VPU_ELEMWISE_FLOAT", expected_bits, actual,
                              ELEM_COUNT);
        return -1;
    }

    return 0;
}

int
main(void)
{
    const uint32_t int_lhs[ELEM_COUNT] = {10U, 20U, 30U, 40U};
    const uint32_t int_rhs[ELEM_COUNT] = {1U, 2U, 3U, 4U};
    const uint32_t int_div_lhs[ELEM_COUNT] = {
        (uint32_t)INT32_MIN, 21U, 9U, 0U
    };
    const uint32_t int_div_rhs[ELEM_COUNT] = {
        (uint32_t)-1, 0U, (uint32_t)-4, 5U
    };
    const uint32_t fp_lhs[ELEM_COUNT] = {
        npu_float_to_bits(1.5f),
        npu_float_to_bits(-2.0f),
        npu_float_to_bits(4.0f),
        npu_float_to_bits(0.5f),
    };
    const uint32_t fp_rhs[ELEM_COUNT] = {
        npu_float_to_bits(2.0f),
        npu_float_to_bits(0.5f),
        npu_float_to_bits(-1.0f),
        npu_float_to_bits(8.0f),
    };
    const uint32_t fp_div_lhs[ELEM_COUNT] = {
        npu_float_to_bits(9.0f),
        npu_float_to_bits(-3.0f),
        npu_float_to_bits(1.0f),
        npu_float_to_bits(0.0f),
    };
    const uint32_t fp_div_rhs[ELEM_COUNT] = {
        npu_float_to_bits(3.0f),
        npu_float_to_bits(0.0f),
        npu_float_to_bits(0.0f),
        npu_float_to_bits(4.0f),
    };
    uint32_t expected[ELEM_COUNT];
    struct VpuStatsExpectation stats = {0};

    if (run_int_case(VPU_OP_VADD, INT_ADD_SYNC, int_lhs, int_rhs, expected,
                     1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_int_case(VPU_OP_VSUB, INT_SUB_SYNC, int_lhs, int_rhs, expected,
                     2U, SRC0_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 2U);

    if (run_int_case(VPU_OP_VMUL, INT_MUL_SYNC, int_lhs, int_rhs, expected,
                     1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_int_case(VPU_OP_VDIV, INT_DIV_SYNC, int_div_lhs, int_div_rhs,
                     expected, 1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_float_case(VPU_OP_VADD, FP_ADD_SYNC, fp_lhs, fp_rhs, expected,
                       1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_float_case(VPU_OP_VSUB, FP_SUB_SYNC, fp_lhs, fp_rhs, expected,
                       1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_float_case(VPU_OP_VMUL, FP_MUL_SYNC, fp_lhs, fp_rhs, expected,
                       2U, SRC0_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 2U);

    if (run_float_case(VPU_OP_VDIV, FP_DIV_SYNC, fp_div_lhs, fp_div_rhs,
                       expected, 1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    printf("VPU_ELEMWISE_EXPECTED_COMPLETED_CMDS=%u\n",
           stats.completed_cmds);
    printf("VPU_ELEMWISE_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_ELEMWISE_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_ELEMWISE_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_ELEMWISE_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_ELEMWISE_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_ELEMWISE_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_ELEMWISE_PASS\n");
    return 0;
}
