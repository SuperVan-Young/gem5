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

enum ReduceSumI32HwPipelineLayout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x92U,
};

int
main(void)
{
    const int32_t src_i32[ELEM_COUNT] = {2, -3, 4, 7};
    uint32_t src_bits[ELEM_COUNT];
    int32_t expected_i32 = 0;
    uint32_t expected = 0;
    uint32_t shape = 0U;
    uint32_t stride = 0U;
    uint32_t w_layout_log2 = 0U;
    uint32_t c_layout_log2 = 0U;
    uint32_t layout_order = 0U;
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        memcpy(&src_bits[idx], &src_i32[idx], sizeof(src_bits[idx]));
    }

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_reduce_sum_i32(src_i32, &expected_i32, ELEM_COUNT);
    memcpy(&expected, &expected_i32, sizeof(expected));

    vpu_default_tensor_geometry(ELEM_COUNT, VPU_DATA_I32, &shape, &stride,
                                &w_layout_log2, &c_layout_log2,
                                &layout_order);
    vpu_cmd_launch_load_one(VPU_DEVICE_ID, 0U, SRC_PORT, ELEM_COUNT,
                            sizeof(uint32_t), VPU_DATA_I32,
                            VPU_DEFAULT_INPUT0_BUFFER);
    {
        const VpuTensorDesc dst = vpu_tensor_desc(
            vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER),
            vpu_pack_shape_field(1U, 1U), vpu_pack_stride_field(1U, 1U));
        const VpuTensorDesc src0 = vpu_tensor_desc(
            vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT0_BUFFER),
            shape, stride);
        const VpuTensorDesc src1 = {0U, 0U, 0U};
        vpu_cmd_launch_compute(VPU_DEVICE_ID, VPU_OP_VREDUCE_SUM, 0U,
                               VPU_DATA_I32, VPU_DATA_I32, VPU_DATA_I32,
                               w_layout_log2, c_layout_log2, layout_order, &dst,
                               &src0, &src1, 0U);
    }
    vpu_cmd_launch_store_one(VPU_DEVICE_ID, SYNC_INDICATOR, DST_PORT, 1U,
                             sizeof(uint32_t), VPU_DATA_I32,
                             VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER);

    if (npu_wait_u32_scalar_match(NULL,
                                  npu_spm_slot_word_ptr_default(DST_PORT),
                                  expected, 60000000ULL) != 0) {
        printf("VPU_REDUCE_SUM_I32_HW_PIPELINE_FAIL exp=%d act=%d\n",
               expected_i32,
               (int32_t)npu_spm_slot_word_ptr_default(DST_PORT)[0]);
        return 1;
    }

    printf("VPU_REDUCE_SUM_I32_HW_PIPELINE_PASS\n");
    return 0;
}
