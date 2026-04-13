/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>

#include "cmd/vpu.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"

enum LaunchCostLayout
{
    VPU_DEVICE_ID = 0U,
    DATA_SLOT = 0U,
    LOOP_COUNT = 256U,
    ELEM_COUNT = 1U,
    SYNC_INDICATOR = 0x4EU,
};

static inline VpuTensorDesc
spm_vector_desc(uint32_t slot)
{
    return vpu_tensor_desc(NPU_MEM_DEFAULT_SPM_BASE_ADDR +
                               slot * NPU_MEM_DEFAULT_SPM_SLOT_STRIDE_BYTES,
                           vpu_pack_shape_field(1U, ELEM_COUNT),
                           vpu_pack_stride_field(ELEM_COUNT, 1U));
}

int
main(void)
{
    const uint32_t init[ELEM_COUNT] = {
        npu_float_to_bits(1.0f),
    };
    uint32_t actual[ELEM_COUNT] = {};
    const VpuTensorDesc tensor = spm_vector_desc(DATA_SLOT);
    const VpuTensorDesc zero = vpu_tensor_desc(0U, 0U, 0U);

    npu_spm_clear_slot(DATA_SLOT);
    npu_spm_store_u32_vector(DATA_SLOT, init, ELEM_COUNT);

    for (uint32_t i = 0U; i < LOOP_COUNT; ++i) {
        NpuCmd cmd;
        const uint32_t sync_indicator =
            i + 1U == LOOP_COUNT ? SYNC_INDICATOR : 0U;
        vpu_cmd_init_compute(&cmd, VPU_DEVICE_ID, VPU_OP_VSCALE,
                             sync_indicator, VPU_DATA_F32, VPU_DATA_F32,
                             VPU_DATA_F32, 0U, 0U, VPU_LAYOUT_WC, &tensor,
                             &tensor, &zero, npu_float_to_bits(1.0f));
        cmd.launchCmd();
    }

    npu_launch_sync_wait(VPU_DEVICE_ID, SYNC_INDICATOR, 0U, 0U, 0U);
    npu_cmd_sync_done();
    npu_spm_load_u32_vector(DATA_SLOT, actual, ELEM_COUNT);

    if (npu_expect_u32_vector("VPU_LAUNCH_COST_SMOKE", init, actual,
                              ELEM_COUNT) != 0) {
        printf("VPU_LAUNCH_COST_SMOKE_FAIL\n");
        return 1;
    }

    printf("VPU_LAUNCH_COST_SMOKE_CMDS=%u\n", LOOP_COUNT);
    printf("VPU_LAUNCH_COST_SMOKE_PASS\n");
    return 0;
}
