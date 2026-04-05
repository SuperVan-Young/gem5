/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "golden/vpu_softmax.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum SoftmaxLayout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SOFTMAX_SYNC0 = 0x81U,
    SOFTMAX_SYNC1 = 0x82U,
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
accumulate_expected_stats(struct VpuStatsExpectation *stats)
{
    stats->completed_cmds += 1U;
    stats->prologues += 1U;
    stats->executes += 1U;
    stats->epilogues += 1U;
    stats->read_resps += 1U;
    stats->write_resps += 1U;
    stats->iterations += 1U;
}

static int
run_softmax_case(const uint32_t *src, uint32_t sync_indicator,
                 const char *tag)
{
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src, ELEM_COUNT);
    npu_golden_vpu_softmax_f32(src, expected, ELEM_COUNT);

    vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VSOFTMAX, sync_indicator,
                         0x1U, 0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);

    if (npu_wait_float_vector_close(NULL,
                                    npu_spm_slot_word_ptr_default(DST_PORT),
                                    expected, ELEM_COUNT, 80000000ULL, 0.03f,
                                    0.03f) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_SOFTMAX_FAIL %s\n", tag);
        npu_expect_float_vector_close("VPU_SOFTMAX", expected, actual,
                                      ELEM_COUNT, 0.03f, 0.03f);
        return -1;
    }

    return 0;
}

int
main(void)
{
    const uint32_t src0[ELEM_COUNT] = {
        0x3f800000U, 0x40000000U, 0x40400000U, 0x40800000U,
    };
    const uint32_t src1[ELEM_COUNT] = {
        0x40000000U, 0x40000000U, 0x40000000U, 0x40000000U,
    };
    struct VpuStatsExpectation stats = {0};

    if (run_softmax_case(src0, SOFTMAX_SYNC0, "ramp") != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats);

    if (run_softmax_case(src1, SOFTMAX_SYNC1, "flat") != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats);

    printf("VPU_SOFTMAX_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_SOFTMAX_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_SOFTMAX_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_SOFTMAX_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_SOFTMAX_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_SOFTMAX_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_SOFTMAX_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_SOFTMAX_PASS\n");
    return 0;
}
