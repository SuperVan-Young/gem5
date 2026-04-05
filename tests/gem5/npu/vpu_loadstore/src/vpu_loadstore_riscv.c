/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "vpu.hh"

enum LoadStoreLayout
{
    VPU_DEVICE_ID = 0U,
    PORT0 = 0U,
    PORT1 = 1U,
    ELEM_COUNT = 4U,
    VLOAD_SYNC = 0x81U,
    VSTORE_SYNC = 0x82U,
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

int
main(void)
{
    const uint32_t src0[ELEM_COUNT] = {11U, 22U, 33U, 44U};
    const uint32_t src1[ELEM_COUNT] = {101U, 202U, 303U, 404U};
    const uint32_t poison0[ELEM_COUNT] = {0xdead0001U, 0xdead0002U,
                                          0xdead0003U, 0xdead0004U};
    const uint32_t poison1[ELEM_COUNT] = {0xbeef0001U, 0xbeef0002U,
                                          0xbeef0003U, 0xbeef0004U};
    uint32_t actual[ELEM_COUNT];
    struct VpuStatsExpectation stats = {2U, 2U, 2U, 2U, 2U, 2U, 2U};

    npu_spm_store_u32_vector(PORT0, src0, ELEM_COUNT);
    npu_spm_store_u32_vector(PORT1, src1, ELEM_COUNT);

    vpu_cmd_launch_load(VPU_DEVICE_ID, VLOAD_SYNC, 0x3U, ELEM_COUNT,
                        sizeof(uint32_t), VPU_DATA_I32);
    npu_launch_sync_wait(VPU_DEVICE_ID, VLOAD_SYNC, 0U, 0U, 0U);
    npu_cmd_sync_done();

    npu_spm_store_u32_vector(PORT0, poison0, ELEM_COUNT);
    npu_spm_store_u32_vector(PORT1, poison1, ELEM_COUNT);

    vpu_cmd_launch_store(VPU_DEVICE_ID, VSTORE_SYNC, 0x3U, ELEM_COUNT,
                         sizeof(uint32_t), VPU_DATA_I32);

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(PORT0),
                                  src0, ELEM_COUNT, 60000000ULL) != 0) {
        npu_spm_load_u32_vector(PORT0, actual, ELEM_COUNT);
        printf("VPU_LOADSTORE_FAIL port0\n");
        npu_expect_u32_vector("VPU_LOADSTORE_PORT0", src0, actual, ELEM_COUNT);
        return 1;
    }

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(PORT1),
                                  src1, ELEM_COUNT, 60000000ULL) != 0) {
        npu_spm_load_u32_vector(PORT1, actual, ELEM_COUNT);
        printf("VPU_LOADSTORE_FAIL port1\n");
        npu_expect_u32_vector("VPU_LOADSTORE_PORT1", src1, actual, ELEM_COUNT);
        return 1;
    }

    printf("VPU_LOADSTORE_EXPECTED_COMPLETED_CMDS=%u\n", stats.completed_cmds);
    printf("VPU_LOADSTORE_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_LOADSTORE_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_LOADSTORE_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_LOADSTORE_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_LOADSTORE_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_LOADSTORE_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_LOADSTORE_PASS\n");
    return 0;
}
