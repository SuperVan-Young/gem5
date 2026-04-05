/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum DecodeSmokeLayout
{
    VPU_DEVICE_ID = 0U,
    LEGACY_SYNC = 0x31U,
    VADD_SYNC = 0x32U,
    SRC0_PORT = 1U,
    SRC1_PORT = 2U,
    DST_PORT = 3U,
    ELEM_COUNT = 4U,
};

static uint32_t
legacy_mix(uint32_t current, uint32_t signature, uint32_t iteration,
           uint32_t port_id)
{
    return current + signature + ((iteration + 1U) * 0x10U) + (port_id + 1U);
}

int
main(void)
{
    const uint32_t legacy_seed = 0x01020304U;
    const uint32_t src0[ELEM_COUNT] = {1U, 3U, 5U, 7U};
    const uint32_t src1[ELEM_COUNT] = {2U, 4U, 6U, 8U};
    uint32_t expected_vadd[ELEM_COUNT];
    uint32_t actual_vadd[ELEM_COUNT];

    npu_spm_clear_slot(0U);
    npu_spm_clear_slot(SRC0_PORT);
    npu_spm_clear_slot(SRC1_PORT);
    npu_spm_clear_slot(DST_PORT);

    npu_spm_store_u32_vector(0U, &legacy_seed, 1U);
    vpu_cmd_launch_legacy_exec(VPU_DEVICE_ID, LEGACY_SYNC, 0x1U, 0x1U, 1U);

    if (npu_wait_u32_scalar_match(NULL, npu_spm_slot_word_ptr_default(0U),
                                  legacy_mix(legacy_seed, legacy_seed, 0U, 0U),
                                  60000000ULL) != 0) {
        printf("VPU_DECODE_SMOKE_FAIL legacy actual=%#x\n",
               npu_spm_slot_word_ptr_default(0U)[0]);
        return 1;
    }

    npu_spm_store_u32_vector(SRC0_PORT, src0, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC1_PORT, src1, ELEM_COUNT);

    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        expected_vadd[idx] = src0[idx] + src1[idx];
    }

    vpu_cmd_launch_binary(VPU_DEVICE_ID, VPU_OP_VADD, VADD_SYNC, 0x6U, 0x8U,
                          1U, ELEM_COUNT, sizeof(uint32_t),
                          sizeof(uint32_t), VPU_DATA_I32);

    if (npu_wait_u32_vector_match(NULL, npu_spm_slot_word_ptr_default(DST_PORT),
                                  expected_vadd, ELEM_COUNT,
                                  60000000ULL) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual_vadd, ELEM_COUNT);
        printf("VPU_DECODE_SMOKE_FAIL vector\n");
        npu_expect_u32_vector("VPU_DECODE_SMOKE", expected_vadd, actual_vadd,
                              ELEM_COUNT);
        return 1;
    }

    printf("VPU_DECODE_SMOKE_PASS\n");
    return 0;
}
