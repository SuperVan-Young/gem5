/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmd/common.hh"
#include "cmd/dma.hh"
#include "npu_sync.hh"
#include "vpu.hh"

enum SystemMultiportLayout
{
    SYSTEM_MULTIPORT_NUM_CPUS = 2U,
    SYSTEM_MULTIPORT_SLOT_STRIDE_BYTES = 0x40U,
    SYSTEM_MULTIPORT_DMA_DEVICE_ID = 0x0U,
    SYSTEM_MULTIPORT_VPU0_ID = 0x0U,
    SYSTEM_MULTIPORT_VPU1_ID = 0x1U,
    SYSTEM_MULTIPORT_DMA_SYNC = 0x61U,
    SYSTEM_MULTIPORT_VPU0_SYNC = 0x62U,
    SYSTEM_MULTIPORT_VPU1_SYNC = 0x63U,
    SYSTEM_MULTIPORT_COPYBACK_SYNC = 0x64U,
    SYSTEM_MULTIPORT_DRAM_SRC_BASE = 0x20001000U,
    SYSTEM_MULTIPORT_DRAM_DST_BASE = 0x20002000U,
    SYSTEM_MULTIPORT_MAILBOX_BASE = 0x20000000U,
    SYSTEM_MULTIPORT_MAILBOX_READY = 0x0U,
    SYSTEM_MULTIPORT_MAILBOX_ENQUEUED = 0x4U,
    SYSTEM_MULTIPORT_SPM_BASE = 0x60000000U,
    SYSTEM_MULTIPORT_SRC_SLOT = 0U,
    SYSTEM_MULTIPORT_LINEAR_SLOT = 1U,
    SYSTEM_MULTIPORT_SOFTMAX_SLOT = 2U,
    SYSTEM_MULTIPORT_ELEM_COUNT = 4U,
    SYSTEM_MULTIPORT_VECTOR_BYTES = SYSTEM_MULTIPORT_ELEM_COUNT *
        sizeof(uint32_t),
};

static volatile uint32_t *
spm_slot_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        SYSTEM_MULTIPORT_SPM_BASE +
        ((uint64_t)port_id * SYSTEM_MULTIPORT_SLOT_STRIDE_BYTES));
}

static volatile uint8_t *
dram_src_byte_ptr(uint32_t offset)
{
    return (volatile uint8_t *)(uintptr_t)(
        SYSTEM_MULTIPORT_DRAM_SRC_BASE + offset
    );
}

static volatile uint8_t *
dram_dst_byte_ptr(uint32_t offset)
{
    return (volatile uint8_t *)(uintptr_t)(
        SYSTEM_MULTIPORT_DRAM_DST_BASE + offset
    );
}

static volatile uint32_t *
mailbox_ptr(uint32_t offset)
{
    return (volatile uint32_t *)(uintptr_t)(
        SYSTEM_MULTIPORT_MAILBOX_BASE + offset
    );
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
         idx < SYSTEM_MULTIPORT_SLOT_STRIDE_BYTES / sizeof(uint32_t); ++idx) {
        base[idx] = 0U;
    }
}

static void
clear_dram_dst(void)
{
    for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_VECTOR_BYTES; ++idx) {
        dram_dst_byte_ptr(idx)[0] = 0U;
    }
}

static void
seed_dram_source(const uint32_t *values)
{
    for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_ELEM_COUNT; ++idx) {
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
    for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_ELEM_COUNT; ++idx) {
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
    for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_ELEM_COUNT; ++idx) {
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
    for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_ELEM_COUNT; ++idx) {
        printf(" [%u]=%#x", idx, values[idx]);
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
        SYSTEM_MULTIPORT_VECTOR_BYTES,
        SYSTEM_MULTIPORT_VECTOR_BYTES,
        SYSTEM_MULTIPORT_VECTOR_BYTES,
        1U,
        0U,
        DMA_CUT_DIM_W,
    };

    dma_cmd_launch_move_layout_at(
        port_base, SYSTEM_MULTIPORT_DMA_DEVICE_ID, src_base, dst_base,
        &layout, &layout, sync_idx, 1U);
}

static void
compute_expected(const uint32_t *src, uint32_t *linear, uint32_t *softmax)
{
    static const uint32_t softmax_expected[SYSTEM_MULTIPORT_ELEM_COUNT] = {
        0x3dcff243U, // ~0.101536
        0x3e2b6c44U, // ~0.167405
        0x3e8d5075U, // ~0.276004
        0x3ee8fcd9U, // ~0.455054
    };

    for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_ELEM_COUNT; ++idx) {
        linear[idx] = float_to_bits(bits_to_float(src[idx]) * 0.5f);
        softmax[idx] = softmax_expected[idx];
    }
}

static int
cpu0_main(uint64_t port_base)
{
    const uint32_t src[SYSTEM_MULTIPORT_ELEM_COUNT] = {
        float_to_bits(-1.0f),
        float_to_bits(0.0f),
        float_to_bits(1.0f),
        float_to_bits(2.0f),
    };
    uint32_t linear_expected[SYSTEM_MULTIPORT_ELEM_COUNT];
    uint32_t softmax_expected[SYSTEM_MULTIPORT_ELEM_COUNT];
    uint32_t actual[SYSTEM_MULTIPORT_ELEM_COUNT];

    *mailbox_ptr(SYSTEM_MULTIPORT_MAILBOX_READY) = 0U;
    *mailbox_ptr(SYSTEM_MULTIPORT_MAILBOX_ENQUEUED) = 0U;

    clear_slot(SYSTEM_MULTIPORT_SRC_SLOT);
    clear_slot(SYSTEM_MULTIPORT_LINEAR_SLOT);
    clear_slot(SYSTEM_MULTIPORT_SOFTMAX_SLOT);
    clear_dram_dst();
    seed_dram_source(src);
    compute_expected(src, linear_expected, softmax_expected);

    push_dma_at(port_base, SYSTEM_MULTIPORT_DMA_SYNC,
                SYSTEM_MULTIPORT_DRAM_SRC_BASE, SYSTEM_MULTIPORT_SPM_BASE);
    npu_launch_sync_wait_at(SYSTEM_MULTIPORT_DMA_DEVICE_ID,
                            SYSTEM_MULTIPORT_DMA_SYNC, 0U, 0U, 0U, port_base);
    vpu_cmd_launch_scale_at(port_base, SYSTEM_MULTIPORT_VPU0_ID,
                            SYSTEM_MULTIPORT_VPU0_SYNC, 0x1U, 0x2U, 1U,
                            SYSTEM_MULTIPORT_ELEM_COUNT, sizeof(uint32_t),
                            sizeof(uint32_t), VPU_DATA_F32,
                            float_to_bits(0.5f));
    npu_launch_sync_wait_at(SYSTEM_MULTIPORT_VPU0_ID,
                            SYSTEM_MULTIPORT_VPU0_SYNC, 0U, 0U, 0U,
                            port_base);

    *mailbox_ptr(SYSTEM_MULTIPORT_MAILBOX_READY) = 1U;
    while (*mailbox_ptr(SYSTEM_MULTIPORT_MAILBOX_ENQUEUED) == 0U) {
        asm volatile("" ::: "memory");
    }

    npu_cmd_sync_done_at(port_base);

    if (float_vector_close_spm(SYSTEM_MULTIPORT_LINEAR_SLOT,
                               linear_expected, 0.0001f, 0.0001f) != 0) {
        for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_ELEM_COUNT; ++idx) {
            actual[idx] = spm_slot_ptr(SYSTEM_MULTIPORT_LINEAR_SLOT)[idx];
        }
        print_vector("SYSTEM_MULTIPORT_LINEAR_EXPECTED", linear_expected);
        print_vector("SYSTEM_MULTIPORT_LINEAR_ACTUAL", actual);
        printf("SYSTEM_MULTIPORT_TEST_FAIL\n");
        return 1;
    }

    if (float_vector_close_spm(SYSTEM_MULTIPORT_SOFTMAX_SLOT,
                               softmax_expected, 0.03f, 0.03f) != 0) {
        for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_ELEM_COUNT; ++idx) {
            actual[idx] = spm_slot_ptr(SYSTEM_MULTIPORT_SOFTMAX_SLOT)[idx];
        }
        print_vector("SYSTEM_MULTIPORT_SOFTMAX_EXPECTED", softmax_expected);
        print_vector("SYSTEM_MULTIPORT_SOFTMAX_ACTUAL", actual);
        printf("SYSTEM_MULTIPORT_TEST_FAIL\n");
        return 1;
    }

    if (float_vector_close_dram(softmax_expected, 0.03f, 0.03f) != 0) {
        for (uint32_t idx = 0U; idx < SYSTEM_MULTIPORT_ELEM_COUNT; ++idx) {
            actual[idx] = read_word_le(
                dram_dst_byte_ptr(idx * sizeof(uint32_t)));
        }
        print_vector("SYSTEM_MULTIPORT_DRAM_EXPECTED", softmax_expected);
        print_vector("SYSTEM_MULTIPORT_DRAM_ACTUAL", actual);
        printf("SYSTEM_MULTIPORT_TEST_FAIL\n");
        return 1;
    }

    print_vector("SYSTEM_MULTIPORT_LINEAR_FINAL", linear_expected);
    print_vector("SYSTEM_MULTIPORT_SOFTMAX_FINAL", softmax_expected);
    printf("SYSTEM_MULTIPORT_TEST_PASS\n");
    return 0;
}

static int
cpu1_main(uint64_t port_base)
{
    while (*mailbox_ptr(SYSTEM_MULTIPORT_MAILBOX_READY) == 0U) {
        asm volatile("" ::: "memory");
    }

    vpu_cmd_launch_unary_at(port_base, SYSTEM_MULTIPORT_VPU1_ID,
                            VPU_OP_VSOFTMAX, SYSTEM_MULTIPORT_VPU1_SYNC,
                            0x2U, 0x4U, 1U, SYSTEM_MULTIPORT_ELEM_COUNT,
                            sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32);
    npu_launch_sync_wait_at(SYSTEM_MULTIPORT_VPU1_ID,
                            SYSTEM_MULTIPORT_VPU1_SYNC, 0U, 0U, 0U, port_base);
    push_dma_at(port_base, SYSTEM_MULTIPORT_COPYBACK_SYNC,
                SYSTEM_MULTIPORT_SPM_BASE +
                    (SYSTEM_MULTIPORT_SOFTMAX_SLOT *
                     SYSTEM_MULTIPORT_SLOT_STRIDE_BYTES),
                SYSTEM_MULTIPORT_DRAM_DST_BASE);
    npu_launch_sync_wait_at(SYSTEM_MULTIPORT_DMA_DEVICE_ID,
                            SYSTEM_MULTIPORT_COPYBACK_SYNC,
                            0U, 0U, 0U, port_base);

    *mailbox_ptr(SYSTEM_MULTIPORT_MAILBOX_ENQUEUED) = 1U;
    npu_cmd_sync_done_at(port_base);
    return 0;
}

int
main(int argc, char **argv)
{
    const uint64_t port_base =
        NPU_CMD_PORT_BASE + ((uint64_t)atoi(argv[1]) << 20);

    if (argc != 2) {
        return 2;
    }

    if (atoi(argv[1]) == 0) {
        return cpu0_main(port_base);
    }

    return cpu1_main(port_base);
}
