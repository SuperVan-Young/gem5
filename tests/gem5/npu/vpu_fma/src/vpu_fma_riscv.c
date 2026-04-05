/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "golden/vpu_fma.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum FmaLayout
{
    VPU_DEVICE_ID = 0U,
    SRC0_PORT = 0U,
    SRC1_PORT = 1U,
    SRC2_PORT = 2U,
    DST_PORT = 3U,
    ELEM_COUNT = 4U,
    FMA_SYNC = 0x61U,
    FMA_INPLACE_SYNC = 0x62U,
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
    stats->read_resps += 3U * repetition;
    stats->write_resps += repetition;
    stats->iterations += repetition;
}

static int
run_fma_case(uint32_t sync_indicator, const uint32_t *src0,
             const uint32_t *src1, const uint32_t *src2, uint32_t *expected,
             uint32_t repetition, uint32_t dst_port)
{
    uint32_t actual[ELEM_COUNT];
    const uint32_t read_mask = (1U << SRC0_PORT) | (1U << SRC1_PORT) |
                               (1U << SRC2_PORT);
    const uint32_t write_mask = 1U << dst_port;

    npu_spm_clear_slot(SRC0_PORT);
    npu_spm_clear_slot(SRC1_PORT);
    npu_spm_clear_slot(SRC2_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC0_PORT, src0, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC1_PORT, src1, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC2_PORT, src2, ELEM_COUNT);
    npu_golden_vpu_fma_f32(src0, src1, src2, expected, ELEM_COUNT);

    vpu_cmd_launch_ternary(VPU_DEVICE_ID, VPU_OP_VFMA, sync_indicator,
                           read_mask, write_mask, repetition, ELEM_COUNT,
                           sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_u32_vector_match(NULL,
                                  npu_spm_slot_word_ptr_default(dst_port),
                                  expected, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(dst_port, actual, ELEM_COUNT);
        printf("VPU_FMA_FAIL\n");
        npu_expect_u32_vector("VPU_FMA", expected, actual, ELEM_COUNT);
        return -1;
    }

    return 0;
}

int
main(void)
{
    const uint32_t src0[ELEM_COUNT] = {
        npu_float_to_bits(1.0f),
        npu_float_to_bits(-2.0f),
        npu_float_to_bits(0.5f),
        npu_float_to_bits(4.0f),
    };
    const uint32_t src1[ELEM_COUNT] = {
        npu_float_to_bits(3.0f),
        npu_float_to_bits(0.25f),
        npu_float_to_bits(-8.0f),
        npu_float_to_bits(2.0f),
    };
    const uint32_t src2[ELEM_COUNT] = {
        npu_float_to_bits(0.5f),
        npu_float_to_bits(1.0f),
        npu_float_to_bits(2.0f),
        npu_float_to_bits(-1.0f),
    };
    uint32_t expected[ELEM_COUNT];
    struct VpuStatsExpectation stats = {0};

    if (run_fma_case(FMA_SYNC, src0, src1, src2, expected, 1U, DST_PORT) !=
        0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_fma_case(FMA_INPLACE_SYNC, src0, src1, src2, expected, 2U,
                     SRC0_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 2U);

    printf("VPU_FMA_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_FMA_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_FMA_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_FMA_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_FMA_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_FMA_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_FMA_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_FMA_PASS\n");
    return 0;
}
