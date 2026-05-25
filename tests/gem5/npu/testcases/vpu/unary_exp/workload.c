/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "golden/vpu_unary.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "vpu.hh"

enum UnaryExpLayout
{
    VPU_DEVICE_ID = 0U,
    SRC_PORT = 0U,
    DST_PORT = 1U,
    ELEM_COUNT = 4U,
    SYNC_INDICATOR = 0x56U,
};

static uint32_t
compute_log2(uint32_t value)
{
    uint32_t log2 = 0U;
    while ((1U << log2) < value) {
        ++log2;
    }
    return log2;
}

static int
launch_unary_exp(uint32_t dlen_bytes)
{
    if (dlen_bytes == sizeof(uint32_t)) {
        vpu_cmd_launch_unary(VPU_DEVICE_ID, VPU_OP_VEXP, SYNC_INDICATOR, 0x1U,
                             0x2U, 1U, ELEM_COUNT, sizeof(uint32_t),
                             sizeof(uint32_t), VPU_DATA_F32);
        return 0;
    }

    if ((dlen_bytes % sizeof(uint32_t)) != 0U) {
        printf("VPU_UNARY_EXP_INVALID_DLEN=%u\n", dlen_bytes);
        return -1;
    }

    const uint32_t c_extent = dlen_bytes / sizeof(uint32_t);
    if (c_extent == 0U || (ELEM_COUNT % c_extent) != 0U) {
        printf("VPU_UNARY_EXP_INVALID_LAYOUT_DLEN=%u\n", dlen_bytes);
        return -1;
    }

    const uint32_t w_extent = ELEM_COUNT / c_extent;
    const uint32_t shape = vpu_pack_shape_field(w_extent, c_extent);
    const uint32_t stride = vpu_pack_stride_field(1U, 1U);
    const uint32_t c_layout_log2 = compute_log2(c_extent);
    const VpuTensorDesc zero = {0U, 0U, 0U};
    const VpuTensorDesc spm_src = vpu_tensor_desc(
        0x60000000U + (SRC_PORT * VPU_LOCAL_SLOT_STRIDE), shape, stride);
    const VpuTensorDesc spm_dst = vpu_tensor_desc(
        0x60000000U + (DST_PORT * VPU_LOCAL_SLOT_STRIDE), shape, stride);
    const VpuTensorDesc local_src = vpu_tensor_desc(
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT0_BUFFER), shape,
        stride);
    const VpuTensorDesc local_dst = vpu_tensor_desc(
        vpu_local_addr(
            VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER
        ),
        shape, stride);

    vpu_cmd_launch_compute(VPU_DEVICE_ID, VPU_OP_VLOAD, 0U, VPU_DATA_F32,
                           VPU_DATA_F32, VPU_DATA_F32, 0U, c_layout_log2,
                           VPU_LAYOUT_WC, &local_src, &spm_src, &zero, 0U);
    vpu_cmd_launch_compute(VPU_DEVICE_ID, VPU_OP_VEXP, 0U, VPU_DATA_F32,
                           VPU_DATA_F32, VPU_DATA_F32, 0U, c_layout_log2,
                           VPU_LAYOUT_WC, &local_dst, &local_src, &zero, 0U);
    vpu_cmd_launch_compute(VPU_DEVICE_ID, VPU_OP_VSTORE, SYNC_INDICATOR,
                           VPU_DATA_F32, VPU_DATA_F32, VPU_DATA_F32, 0U,
                           c_layout_log2, VPU_LAYOUT_WC, &spm_dst, &local_dst,
                           &zero, 0U);
    return 0;
}

int
main(int argc, char **argv)
{
    const uint32_t dlen_bytes = argc > 1 ? (uint32_t)atoi(argv[1]) :
                                           sizeof(uint32_t);
    const uint32_t src_bits[ELEM_COUNT] = {
        npu_float_to_bits(-1.0f),
        npu_float_to_bits(-0.5f),
        npu_float_to_bits(0.0f),
        npu_float_to_bits(1.0f),
    };
    uint32_t expected[ELEM_COUNT];
    uint32_t actual[ELEM_COUNT];

    npu_spm_clear_slot(SRC_PORT);
    npu_spm_clear_slot(DST_PORT);
    npu_spm_store_u32_vector(SRC_PORT, src_bits, ELEM_COUNT);
    npu_golden_vpu_unary_exp(src_bits, expected, ELEM_COUNT);

    if (launch_unary_exp(dlen_bytes) != 0) {
        printf("VPU_UNARY_EXP_FAIL\n");
        return 1;
    }

    if (npu_wait_float_vector_close(NULL,
                                    npu_spm_slot_word_ptr_default(DST_PORT),
                                    expected, ELEM_COUNT, 60000000ULL, 0.03f,
                                    0.03f) != 0) {
        npu_spm_load_u32_vector(DST_PORT, actual, ELEM_COUNT);
        printf("VPU_UNARY_EXP_FAIL\n");
        npu_expect_float_vector_close("VPU_UNARY_EXP", expected, actual,
                                      ELEM_COUNT, 0.03f, 0.03f);
        return 1;
    }

    printf("VPU_UNARY_EXP_PASS\n");
    return 0;
}
