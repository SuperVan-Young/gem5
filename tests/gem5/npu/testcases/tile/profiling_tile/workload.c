/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cmd/common.hh"
#include "cmd/dma.hh"
#include "cmd/vpu.hh"
#include "golden/vpu_softmax.hh"
#include "npu_sync.hh"

enum ProfileTileLayout
{
    PROFILE_TILE_DMA_DEVICE_ID = 0x0U,
    PROFILE_TILE_VPU0_ID = 0x0U,
    PROFILE_TILE_VPU1_ID = 0x1U,
    PROFILE_TILE_DMA_SYNC = 0x71U,
    PROFILE_TILE_VPU0_SYNC = 0x72U,
    PROFILE_TILE_VPU1_SYNC = 0x73U,
    PROFILE_TILE_COPYBACK_SYNC = 0x74U,
    PROFILE_TILE_DRAM_SRC_BASE = 0x20001000U,
    PROFILE_TILE_DRAM_DST_BASE = 0x20002000U,
    PROFILE_TILE_SPM_BASE = 0x60000000U,
    PROFILE_TILE_SRC_SLOT = 0U,
    PROFILE_TILE_LINEAR_SLOT = 1U,
    PROFILE_TILE_SOFTMAX_SLOT = 2U,
    PROFILE_TILE_SLOT_STRIDE_BYTES = 0x40U,
    PROFILE_TILE_ELEM_COUNT = 16U,
    PROFILE_TILE_VECTOR_BYTES = PROFILE_TILE_ELEM_COUNT * sizeof(uint32_t),
};

static volatile uint32_t *
spm_slot_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        PROFILE_TILE_SPM_BASE +
        ((uint64_t)port_id * PROFILE_TILE_SLOT_STRIDE_BYTES));
}

static volatile uint8_t *
dram_src_byte_ptr(uint32_t offset)
{
    return (volatile uint8_t *)(uintptr_t)(
        PROFILE_TILE_DRAM_SRC_BASE + offset);
}

static volatile uint8_t *
dram_dst_byte_ptr(uint32_t offset)
{
    return (volatile uint8_t *)(uintptr_t)(
        PROFILE_TILE_DRAM_DST_BASE + offset);
}

static uint32_t
float_to_bits(float value)
{
    uint32_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float
bits_to_float(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void
write_word_le(volatile uint8_t *ptr, uint32_t value)
{
    ptr[0] = (uint8_t)(value & 0xFFU);
    ptr[1] = (uint8_t)((value >> 8) & 0xFFU);
    ptr[2] = (uint8_t)((value >> 16) & 0xFFU);
    ptr[3] = (uint8_t)((value >> 24) & 0xFFU);
}

static uint32_t
read_word_le(volatile uint8_t *ptr)
{
    return ((uint32_t)ptr[0]) |
           ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

static void
clear_slot(uint32_t port_id)
{
    volatile uint32_t *base = spm_slot_ptr(port_id);

    for (uint32_t idx = 0U;
         idx < PROFILE_TILE_SLOT_STRIDE_BYTES / sizeof(uint32_t); ++idx) {
        base[idx] = 0U;
    }
}

static void
clear_dram_dst(void)
{
    for (uint32_t idx = 0U; idx < PROFILE_TILE_VECTOR_BYTES; ++idx) {
        dram_dst_byte_ptr(idx)[0] = 0U;
    }
}

static void
seed_dram_source(const uint32_t *values)
{
    for (uint32_t idx = 0U; idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
        write_word_le(dram_src_byte_ptr(idx * sizeof(uint32_t)), values[idx]);
    }
}

static float
float_abs(float value)
{
    return value < 0.0f ? -value : value;
}

static int
float_close(float expected, float actual, float abs_tol, float rel_tol)
{
    const float diff = float_abs(expected - actual);
    const float limit = abs_tol + (rel_tol * float_abs(expected));
    return diff <= limit;
}

static int
float_vector_close_spm(uint32_t port_id, const uint32_t *expected,
                       float abs_tol, float rel_tol)
{
    for (uint32_t idx = 0U; idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
        const float expected_value = bits_to_float(expected[idx]);
        const float actual_value = bits_to_float(spm_slot_ptr(port_id)[idx]);
        if (!float_close(expected_value, actual_value, abs_tol, rel_tol)) {
            return -1;
        }
    }

    return 0;
}

static int
float_vector_close_dram(const uint32_t *expected, float abs_tol, float rel_tol)
{
    for (uint32_t idx = 0U; idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
        const float expected_value = bits_to_float(expected[idx]);
        const float actual_value = bits_to_float(
            read_word_le(dram_dst_byte_ptr(idx * sizeof(uint32_t))));
        if (!float_close(expected_value, actual_value, abs_tol, rel_tol)) {
            return -1;
        }
    }

    return 0;
}

static void
print_vector(const char *prefix, const uint32_t *values)
{
    printf("%s", prefix);
    for (uint32_t idx = 0U;
         idx < PROFILE_TILE_ELEM_COUNT && idx < 8U; ++idx) {
        printf(" [%u]=%#x", idx, values[idx]);
    }
    if (PROFILE_TILE_ELEM_COUNT > 12U) {
        printf(" ...");
        for (uint32_t idx = PROFILE_TILE_ELEM_COUNT - 4U;
             idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
            printf(" [%u]=%#x", idx, values[idx]);
        }
    }
    printf("\n");
}

static void
push_dma_at(uint64_t port_base, uint32_t sync_idx,
            uint32_t src_base, uint32_t dst_base)
{
    const DmaLayout layout = {
        1U,
        1U,
        PROFILE_TILE_VECTOR_BYTES,
        PROFILE_TILE_VECTOR_BYTES,
        PROFILE_TILE_VECTOR_BYTES,
        1U,
        0U,
        DMA_CUT_DIM_W,
    };

    dma_cmd_launch_move_layout_at(
        port_base, PROFILE_TILE_DMA_DEVICE_ID, src_base, dst_base,
        &layout, &layout, sync_idx, 1U);
}

static void
compute_expected(const uint32_t *src, uint32_t *linear, uint32_t *softmax)
{
    for (uint32_t idx = 0U; idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
        linear[idx] = float_to_bits(bits_to_float(src[idx]) * 0.5f);
    }
    npu_golden_vpu_softmax_f32(linear, softmax, PROFILE_TILE_ELEM_COUNT);
}

static void
seed_source_vector(uint32_t *src)
{
    for (uint32_t idx = 0U; idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
        const int32_t centered = static_cast<int32_t>(idx % 32U) - 16;
        const float value =
            (static_cast<float>(centered) * 0.125f) +
            (static_cast<float>(idx / 32U) * 0.03125f);
        src[idx] = float_to_bits(value);
    }
}

int
main(void)
{
    uint32_t src[PROFILE_TILE_ELEM_COUNT];
    uint32_t linear_expected[PROFILE_TILE_ELEM_COUNT];
    uint32_t softmax_expected[PROFILE_TILE_ELEM_COUNT];
    uint32_t actual[PROFILE_TILE_ELEM_COUNT];

    clear_slot(PROFILE_TILE_SRC_SLOT);
    clear_slot(PROFILE_TILE_LINEAR_SLOT);
    clear_slot(PROFILE_TILE_SOFTMAX_SLOT);
    clear_dram_dst();
    seed_source_vector(src);
    seed_dram_source(src);
    compute_expected(src, linear_expected, softmax_expected);

    push_dma_at(NPU_CMD_PORT_BASE, PROFILE_TILE_DMA_SYNC,
                PROFILE_TILE_DRAM_SRC_BASE, PROFILE_TILE_SPM_BASE);
    npu_launch_sync_wait_at(PROFILE_TILE_DMA_DEVICE_ID,
                            PROFILE_TILE_DMA_SYNC, 0U, 0U, 0U,
                            NPU_CMD_PORT_BASE);
    vpu_cmd_launch_scale(PROFILE_TILE_VPU0_ID, PROFILE_TILE_VPU0_SYNC,
                         0x1U, 0x2U, 1U, PROFILE_TILE_ELEM_COUNT,
                         sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32,
                         float_to_bits(0.5f));
    npu_launch_sync_wait_at(PROFILE_TILE_VPU0_ID,
                            PROFILE_TILE_VPU0_SYNC, 0U, 0U, 0U,
                            NPU_CMD_PORT_BASE);
    vpu_cmd_launch_unary(PROFILE_TILE_VPU1_ID, VPU_OP_VSOFTMAX,
                         PROFILE_TILE_VPU1_SYNC, 0x2U, 0x4U, 1U,
                         PROFILE_TILE_ELEM_COUNT, sizeof(uint32_t),
                         sizeof(uint32_t), VPU_DATA_F32);
    npu_launch_sync_wait_at(PROFILE_TILE_VPU1_ID,
                            PROFILE_TILE_VPU1_SYNC, 0U, 0U, 0U,
                            NPU_CMD_PORT_BASE);
    push_dma_at(NPU_CMD_PORT_BASE, PROFILE_TILE_COPYBACK_SYNC,
                PROFILE_TILE_SPM_BASE +
                    (PROFILE_TILE_SOFTMAX_SLOT *
                     PROFILE_TILE_SLOT_STRIDE_BYTES),
                PROFILE_TILE_DRAM_DST_BASE);
    npu_launch_sync_wait_at(PROFILE_TILE_DMA_DEVICE_ID,
                            PROFILE_TILE_COPYBACK_SYNC,
                            0U, 0U, 0U, NPU_CMD_PORT_BASE);

    if (float_vector_close_spm(PROFILE_TILE_LINEAR_SLOT,
                               linear_expected, 0.0001f, 0.0001f) != 0) {
        for (uint32_t idx = 0U; idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
            actual[idx] = spm_slot_ptr(PROFILE_TILE_LINEAR_SLOT)[idx];
        }
        print_vector("PROFILE_TILE_LINEAR_EXPECTED", linear_expected);
        print_vector("PROFILE_TILE_LINEAR_ACTUAL", actual);
        printf("PROFILE_TILE_TEST_FAIL\n");
        return 1;
    }

    if (float_vector_close_spm(PROFILE_TILE_SOFTMAX_SLOT,
                               softmax_expected, 0.03f, 0.03f) != 0) {
        for (uint32_t idx = 0U; idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
            actual[idx] = spm_slot_ptr(PROFILE_TILE_SOFTMAX_SLOT)[idx];
        }
        print_vector("PROFILE_TILE_SOFTMAX_EXPECTED", softmax_expected);
        print_vector("PROFILE_TILE_SOFTMAX_ACTUAL", actual);
        printf("PROFILE_TILE_TEST_FAIL\n");
        return 1;
    }

    if (float_vector_close_dram(softmax_expected, 0.03f, 0.03f) != 0) {
        for (uint32_t idx = 0U; idx < PROFILE_TILE_ELEM_COUNT; ++idx) {
            actual[idx] = read_word_le(
                dram_dst_byte_ptr(idx * sizeof(uint32_t)));
        }
        print_vector("PROFILE_TILE_DRAM_EXPECTED", softmax_expected);
        print_vector("PROFILE_TILE_DRAM_ACTUAL", actual);
        printf("PROFILE_TILE_TEST_FAIL\n");
        return 1;
    }

    print_vector("PROFILE_TILE_LINEAR_FINAL", linear_expected);
    print_vector("PROFILE_TILE_SOFTMAX_FINAL", softmax_expected);
    npu_cmd_sync_done();
    printf("PROFILE_TILE_TEST_PASS\n");
    return 0;
}
