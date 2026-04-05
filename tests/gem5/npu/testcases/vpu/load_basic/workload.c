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

enum LoadBasicLayout
{
    VPU_DEVICE_ID = 0U,
    PORT0 = 0U,
    PORT1 = 1U,
    ELEM_COUNT = 4U,
    LOAD_SYNC = 0x81U,
};

int
main(void)
{
    const uint32_t src0[ELEM_COUNT] = {11U, 22U, 33U, 44U};
    const uint32_t src1[ELEM_COUNT] = {101U, 202U, 303U, 404U};

    npu_spm_store_u32_vector(PORT0, src0, ELEM_COUNT);
    npu_spm_store_u32_vector(PORT1, src1, ELEM_COUNT);

    vpu_cmd_launch_load(VPU_DEVICE_ID, LOAD_SYNC, 0x3U, ELEM_COUNT,
                        sizeof(uint32_t), VPU_DATA_I32);
    npu_launch_sync_wait(VPU_DEVICE_ID, LOAD_SYNC, 0U, 0U, 0U);
    npu_cmd_sync_done();

    printf("VPU_LOAD_BASIC_PASS\n");
    return 0;
}
