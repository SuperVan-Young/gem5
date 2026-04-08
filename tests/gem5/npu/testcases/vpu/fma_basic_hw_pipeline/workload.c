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

enum FmaBasicHwPipelineLayout
{
    VPU_DEVICE_ID = 0U,
    SRC0_PORT = 0U,
    SRC1_PORT = 1U,
    SRC2_PORT = 2U,
    DST_PORT = 3U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x93U,
};

static inline uint32_t
spm_slot_addr(uint32_t port)
{
    return (uint32_t)(NPU_MEM_DEFAULT_SPM_BASE_ADDR +
                      ((uint64_t)port *
                       NPU_MEM_DEFAULT_SPM_SLOT_STRIDE_BYTES));
}

int
main(void)
{
    const uint32_t src0[ELEM_COUNT] = {
        npu_float_to_bits(1.0f),
        npu_float_to_bits(2.0f),
        npu_float_to_bits(3.0f),
        npu_float_to_bits(4.0f),
    };
    const uint32_t src1[ELEM_COUNT] = {
        npu_float_to_bits(5.0f),
        npu_float_to_bits(6.0f),
        npu_float_to_bits(7.0f),
        npu_float_to_bits(8.0f),
    };
    const uint32_t src2[ELEM_COUNT] = {
        npu_float_to_bits(0.5f),
        npu_float_to_bits(1.0f),
        npu_float_to_bits(1.5f),
        npu_float_to_bits(2.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];
    NpuCmd cmd;

    npu_spm_clear_slot(SRC0_PORT);
    npu_spm_clear_slot(SRC1_PORT);
    npu_spm_clear_slot(SRC2_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC0_PORT, src0, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC1_PORT, src1, ELEM_COUNT);
    npu_spm_store_u32_vector(SRC2_PORT, src2, ELEM_COUNT);
    npu_golden_vpu_fma_f32(src0, src1, src2, expected, ELEM_COUNT);

    vpu_cmd_init_raw(&cmd, VPU_DEVICE_ID, VPU_OP_VFMA, SYNC_INDICATOR);
    vpu_cmd_set_common_fields(
        &cmd, 0x7U, 0x8U, 1U, 0U, ELEM_COUNT, sizeof(uint32_t),
        sizeof(uint32_t), VPU_DATA_F32, 0U, spm_slot_addr(SRC0_PORT),
        spm_slot_addr(SRC1_PORT), spm_slot_addr(SRC2_PORT),
        spm_slot_addr(DST_PORT));
    cmd.launchCmd();

    if (npu_wait_float_vector_close(
            NULL, npu_spm_slot_word_ptr_default(DST_PORT), expected,
            ELEM_COUNT, 60000000ULL, 0.03f, 0.03f) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_FMA_BASIC_HW_PIPELINE_FAIL\n");
        npu_expect_float_vector_close("VPU_FMA_BASIC_HW_PIPELINE", expected,
                                      actual, ELEM_COUNT, 0.03f, 0.03f);
        return 1;
    }

    printf("VPU_FMA_BASIC_HW_PIPELINE_PASS\n");
    return 0;
}
