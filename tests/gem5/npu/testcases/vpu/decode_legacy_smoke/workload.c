/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum DecodeLegacySmokeLayout
{
    VPU_DEVICE_ID = 0U,
    LEGACY_SYNC = 0x31U,
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
    const uint32_t expected = legacy_mix(legacy_seed, legacy_seed, 0U, 0U);

    npu_spm_clear_slot(0U);
    npu_spm_store_u32_vector(0U, &legacy_seed, 1U);

    vpu_cmd_launch_legacy_exec(VPU_DEVICE_ID, LEGACY_SYNC, 0x1U, 0x1U, 1U);

    if (npu_wait_u32_scalar_match(NULL, npu_spm_slot_word_ptr_default(0U),
                                  expected, 60000000ULL) != 0) {
        printf("VPU_DECODE_LEGACY_SMOKE_FAIL actual=%#x expected=%#x\n",
               npu_spm_slot_word_ptr_default(0U)[0], expected);
        return 1;
    }

    printf("VPU_DECODE_LEGACY_SMOKE_PASS\n");
    return 0;
}
