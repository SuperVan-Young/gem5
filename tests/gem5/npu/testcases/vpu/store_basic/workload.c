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

enum StoreBasicLayout
{
    VPU_DEVICE_ID = 0U,
    PORT0 = 0U,
    PORT1 = 1U,
    ELEM_COUNT = 4U,
    LOAD_SYNC = 0x81U,
    STORE_SYNC = 0x82U,
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

    npu_spm_store_u32_vector(PORT0, src0, ELEM_COUNT);
    npu_spm_store_u32_vector(PORT1, src1, ELEM_COUNT);

    vpu_cmd_launch_load(VPU_DEVICE_ID, LOAD_SYNC, 0x3U, ELEM_COUNT,
                        sizeof(uint32_t), VPU_DATA_I32);
    npu_launch_sync_wait(VPU_DEVICE_ID, LOAD_SYNC, 0U, 0U, 0U);
    npu_cmd_sync_done();

    npu_spm_store_u32_vector(PORT0, poison0, ELEM_COUNT);
    npu_spm_store_u32_vector(PORT1, poison1, ELEM_COUNT);

    vpu_cmd_launch_store(VPU_DEVICE_ID, STORE_SYNC, 0x3U, ELEM_COUNT,
                         sizeof(uint32_t), VPU_DATA_I32);

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(PORT0),
                                  src0, ELEM_COUNT, 60000000ULL) != 0) {
        npu_spm_load_u32_vector(PORT0, actual, ELEM_COUNT);
        printf("VPU_STORE_BASIC_FAIL port0\n");
        npu_expect_u32_vector("VPU_STORE_BASIC_PORT0", src0, actual,
                              ELEM_COUNT);
        return 1;
    }

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(PORT1),
                                  src1, ELEM_COUNT, 60000000ULL) != 0) {
        npu_spm_load_u32_vector(PORT1, actual, ELEM_COUNT);
        printf("VPU_STORE_BASIC_FAIL port1\n");
        npu_expect_u32_vector("VPU_STORE_BASIC_PORT1", src1, actual,
                              ELEM_COUNT);
        return 1;
    }

    printf("VPU_STORE_BASIC_PASS\n");
    return 0;
}
