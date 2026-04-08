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
    PROFILE_TILE_PROBE_ITERS = 128U,
};

#ifdef PROFILE_TILE_ENABLE_BREAKDOWN_PROBES
struct BuildProbeResult
{
    uint64_t init_cycles;
    uint64_t fill_cycles;
    uint64_t total_cycles;
};

static uint64_t g_cycle_probe_overhead = 0;

static inline uint64_t
read_cycle(void)
{
    uint64_t value = 0;
    asm volatile("rdcycle %0" : "=r"(value));
    return value;
}

static uint64_t
measure_rdcycle_overhead(void)
{
    uint64_t total = 0;

    for (uint32_t iter = 0; iter < PROFILE_TILE_PROBE_ITERS; ++iter) {
        const uint64_t start = read_cycle();
        const uint64_t end = read_cycle();
        total += (end - start);
    }

    return total / PROFILE_TILE_PROBE_ITERS;
}

static inline uint64_t
sample_cycles(uint64_t start, uint64_t end)
{
    const uint64_t delta = end - start;
    return delta > g_cycle_probe_overhead ? delta - g_cycle_probe_overhead : 0;
}

static BuildProbeResult
measure_dma_build_cycles(uint32_t sync_idx, uint32_t src_base, uint32_t dst_base)
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
    uint64_t total_cycles = 0;

    for (uint32_t iter = 0; iter < PROFILE_TILE_PROBE_ITERS; ++iter) {
        NpuCmd cmd;
        const uint64_t start = read_cycle();
        dma_cmd_init_move_layout(&cmd, PROFILE_TILE_DMA_DEVICE_ID, src_base,
                                 dst_base, &layout, &layout, sync_idx, 1U);
        total_cycles += sample_cycles(start, read_cycle());
    }

    return {0U, 0U, total_cycles / PROFILE_TILE_PROBE_ITERS};
}

static BuildProbeResult
measure_sync_build_cycles(uint32_t device_id, uint32_t sync_indicator)
{
    uint64_t total_cycles = 0;

    for (uint32_t iter = 0; iter < PROFILE_TILE_PROBE_ITERS; ++iter) {
        NpuCmd cmd;
        const uint64_t start = read_cycle();
        npuBuildSyncWaitCmd(&cmd, device_id, sync_indicator, 0U, 0U, 0U);
        total_cycles += sample_cycles(start, read_cycle());
    }

    return {0U, 0U, total_cycles / PROFILE_TILE_PROBE_ITERS};
}

static BuildProbeResult
measure_vpu_load_build_cycles(uint32_t device_id, uint32_t port,
                              uint32_t elem_count, uint32_t src_stride_bytes,
                              uint32_t data_type)
{
    uint64_t init_cycles = 0;
    uint64_t fill_cycles = 0;
    uint64_t total_cycles = 0;

    for (uint32_t iter = 0; iter < PROFILE_TILE_PROBE_ITERS; ++iter) {
        NpuCmd cmd;
        uint64_t start = read_cycle();
        const uint64_t total_start = start;

        vpu_cmd_init_raw(&cmd, device_id, VPU_OP_VLOAD, 0U);
        init_cycles += sample_cycles(start, read_cycle());

        start = read_cycle();
        vpu_cmd_set_common_fields(
            &cmd, 1U << port, 0U, 1U, 0U, elem_count, src_stride_bytes, 0U,
            data_type, 0U, 0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE), 0U,
            0U, vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER));
        fill_cycles += sample_cycles(start, read_cycle());
        total_cycles += sample_cycles(total_start, read_cycle());
    }

    return {init_cycles / PROFILE_TILE_PROBE_ITERS,
            fill_cycles / PROFILE_TILE_PROBE_ITERS,
            total_cycles / PROFILE_TILE_PROBE_ITERS};
}

static BuildProbeResult
measure_vpu_compute_build_cycles(uint32_t device_id, uint32_t op_code,
                                 uint32_t read_mask, uint32_t write_mask,
                                 uint32_t repetition, uint32_t elem_count,
                                 uint32_t src_stride_bytes,
                                 uint32_t dst_stride_bytes,
                                 uint32_t data_type, uint32_t scalar_bits)
{
    uint64_t init_cycles = 0;
    uint64_t fill_cycles = 0;
    uint64_t total_cycles = 0;

    for (uint32_t iter = 0; iter < PROFILE_TILE_PROBE_ITERS; ++iter) {
        NpuCmd cmd;
        uint64_t start = read_cycle();
        const uint64_t total_start = start;

        vpu_cmd_init_raw(&cmd, device_id, op_code, 0U);
        init_cycles += sample_cycles(start, read_cycle());

        start = read_cycle();
        vpu_cmd_set_common_fields(
            &cmd, read_mask, write_mask, repetition, 0U, elem_count,
            src_stride_bytes, dst_stride_bytes, data_type, scalar_bits,
            vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
            vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
            vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
            vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER));
        fill_cycles += sample_cycles(start, read_cycle());
        total_cycles += sample_cycles(total_start, read_cycle());
    }

    return {init_cycles / PROFILE_TILE_PROBE_ITERS,
            fill_cycles / PROFILE_TILE_PROBE_ITERS,
            total_cycles / PROFILE_TILE_PROBE_ITERS};
}

static BuildProbeResult
measure_vpu_store_build_cycles(uint32_t device_id, uint32_t sync_indicator,
                               uint32_t port, uint32_t elem_count,
                               uint32_t dst_stride_bytes, uint32_t data_type)
{
    uint64_t init_cycles = 0;
    uint64_t fill_cycles = 0;
    uint64_t total_cycles = 0;

    for (uint32_t iter = 0; iter < PROFILE_TILE_PROBE_ITERS; ++iter) {
        NpuCmd cmd;
        uint64_t start = read_cycle();
        const uint64_t total_start = start;

        vpu_cmd_init_raw(&cmd, device_id, VPU_OP_VSTORE, sync_indicator);
        init_cycles += sample_cycles(start, read_cycle());

        start = read_cycle();
        vpu_cmd_set_common_fields(
            &cmd, 0U, 1U << port, 1U, 0U, elem_count, 0U, dst_stride_bytes,
            data_type, 0U,
            vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER),
            0U, 0U, 0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE));
        fill_cycles += sample_cycles(start, read_cycle());
        total_cycles += sample_cycles(total_start, read_cycle());
    }

    return {init_cycles / PROFILE_TILE_PROBE_ITERS,
            fill_cycles / PROFILE_TILE_PROBE_ITERS,
            total_cycles / PROFILE_TILE_PROBE_ITERS};
}

static uint64_t
measure_stage_cycles(const NpuCmd *cmd)
{
    uint64_t total_cycles = 0;

    for (uint32_t iter = 0; iter < PROFILE_TILE_PROBE_ITERS; ++iter) {
        const uint64_t start = read_cycle();
        cmd->stageCmdWords();
        total_cycles += sample_cycles(start, read_cycle());
    }

    return total_cycles / PROFILE_TILE_PROBE_ITERS;
}
#endif

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
issue_profiled_dma_move(uint32_t sync_idx, uint32_t src_base, uint32_t dst_base)
{
    static NpuCmd cmd;
    static const DmaLayout layout = {
        1U,
        1U,
        PROFILE_TILE_VECTOR_BYTES,
        PROFILE_TILE_VECTOR_BYTES,
        PROFILE_TILE_VECTOR_BYTES,
        1U,
        0U,
        DMA_CUT_DIM_W,
    };

    dma_cmd_init_move_layout(
        &cmd, PROFILE_TILE_DMA_DEVICE_ID, src_base, dst_base, &layout, &layout,
        sync_idx, 1U);
    cmd.stageCmdWords();
    cmd.launchStagedCmdAt(NPU_CMD_PORT_BASE);
}

static void
issue_profiled_sync_wait(uint32_t device_id, uint32_t sync_indicator)
{
    static NpuCmd cmd;

    npuBuildSyncWaitCmd(&cmd, device_id, sync_indicator, 0U, 0U, 0U);
    cmd.stageCmdWords();
    cmd.launchStagedCmdAt(NPU_CMD_PORT_BASE);
}

static void
issue_profiled_vpu_load(uint32_t device_id, uint32_t port, uint32_t elem_count,
                        uint32_t src_stride_bytes, uint32_t data_type)
{
    static NpuCmd cmd;

    vpu_cmd_init_raw(&cmd, device_id, VPU_OP_VLOAD, 0U);
    vpu_cmd_set_common_fields(
        &cmd, 1U << port, 0U, 1U, 0U, elem_count, src_stride_bytes, 0U,
        data_type, 0U, 0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE), 0U, 0U,
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER));
    cmd.stageCmdWords();
    cmd.launchStagedCmdAt(NPU_CMD_PORT_BASE);
}

static void
issue_profiled_vpu_compute(uint32_t device_id, uint32_t op_code,
                           uint32_t read_mask,
                           uint32_t write_mask, uint32_t repetition,
                           uint32_t elem_count, uint32_t src_stride_bytes,
                           uint32_t dst_stride_bytes, uint32_t data_type,
                           uint32_t scalar_bits)
{
    static NpuCmd cmd;

    vpu_cmd_init_raw(&cmd, device_id, op_code, 0U);
    vpu_cmd_set_common_fields(
        &cmd, read_mask, write_mask, repetition, 0U, elem_count,
        src_stride_bytes, dst_stride_bytes, data_type, scalar_bits,
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER));
    cmd.stageCmdWords();
    cmd.launchStagedCmdAt(NPU_CMD_PORT_BASE);
}

static void
issue_profiled_vpu_store(uint32_t device_id, uint32_t sync_indicator,
                         uint32_t port,
                         uint32_t elem_count, uint32_t dst_stride_bytes,
                         uint32_t data_type)
{
    static NpuCmd cmd;

    vpu_cmd_init_raw(&cmd, device_id, VPU_OP_VSTORE, sync_indicator);
    vpu_cmd_set_common_fields(
        &cmd, 0U, 1U << port, 1U, 0U, elem_count, 0U, dst_stride_bytes,
        data_type, 0U,
        vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER),
        0U, 0U, 0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE));
    cmd.stageCmdWords();
    cmd.launchStagedCmdAt(NPU_CMD_PORT_BASE);
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
#ifdef PROFILE_TILE_ENABLE_BREAKDOWN_PROBES
    const DmaLayout dma_layout = {
        1U,
        1U,
        PROFILE_TILE_VECTOR_BYTES,
        PROFILE_TILE_VECTOR_BYTES,
        PROFILE_TILE_VECTOR_BYTES,
        1U,
        0U,
        DMA_CUT_DIM_W,
    };
#endif
    uint32_t src[PROFILE_TILE_ELEM_COUNT];
    uint32_t linear_expected[PROFILE_TILE_ELEM_COUNT];
    uint32_t softmax_expected[PROFILE_TILE_ELEM_COUNT];
    uint32_t actual[PROFILE_TILE_ELEM_COUNT];
#ifdef PROFILE_TILE_ENABLE_BREAKDOWN_PROBES
    NpuCmd dma_probe_cmd;
    NpuCmd sync_probe_cmd;
    NpuCmd vpu_load_probe_cmd;
    NpuCmd vpu_compute_probe_cmd;
    NpuCmd vpu_store_probe_cmd;
    BuildProbeResult dma_build = {};
    BuildProbeResult sync_build = {};
    BuildProbeResult vpu_load_build = {};
    BuildProbeResult vpu_compute_build = {};
    BuildProbeResult vpu_store_build = {};
#endif

    clear_slot(PROFILE_TILE_SRC_SLOT);
    clear_slot(PROFILE_TILE_LINEAR_SLOT);
    clear_slot(PROFILE_TILE_SOFTMAX_SLOT);
    clear_dram_dst();
    seed_source_vector(src);
    seed_dram_source(src);
    compute_expected(src, linear_expected, softmax_expected);
#ifdef PROFILE_TILE_ENABLE_BREAKDOWN_PROBES
    g_cycle_probe_overhead = measure_rdcycle_overhead();
    dma_build = measure_dma_build_cycles(
        PROFILE_TILE_DMA_SYNC, PROFILE_TILE_DRAM_SRC_BASE,
        PROFILE_TILE_SPM_BASE);
    sync_build = measure_sync_build_cycles(
        PROFILE_TILE_DMA_DEVICE_ID, PROFILE_TILE_DMA_SYNC);
    vpu_load_build = measure_vpu_load_build_cycles(
        PROFILE_TILE_VPU0_ID, PROFILE_TILE_SRC_SLOT, PROFILE_TILE_ELEM_COUNT,
        sizeof(uint32_t), VPU_DATA_F32);
    vpu_compute_build = measure_vpu_compute_build_cycles(
        PROFILE_TILE_VPU0_ID, VPU_OP_VSCALE, 0x1U, 0x2U, 1U,
        PROFILE_TILE_ELEM_COUNT, sizeof(uint32_t), sizeof(uint32_t),
        VPU_DATA_F32, float_to_bits(0.5f));
    vpu_store_build = measure_vpu_store_build_cycles(
        PROFILE_TILE_VPU0_ID, PROFILE_TILE_VPU0_SYNC,
        PROFILE_TILE_LINEAR_SLOT, PROFILE_TILE_ELEM_COUNT, sizeof(uint32_t),
        VPU_DATA_F32);
    dma_cmd_init_move_layout(
        &dma_probe_cmd, PROFILE_TILE_DMA_DEVICE_ID, PROFILE_TILE_DRAM_SRC_BASE,
        PROFILE_TILE_SPM_BASE, &dma_layout, &dma_layout, PROFILE_TILE_DMA_SYNC,
        1U);
    npuBuildSyncWaitCmd(&sync_probe_cmd, PROFILE_TILE_DMA_DEVICE_ID,
                        PROFILE_TILE_DMA_SYNC, 0U, 0U, 0U);
    vpu_cmd_init_raw(&vpu_load_probe_cmd, PROFILE_TILE_VPU0_ID, VPU_OP_VLOAD,
                     0U);
    vpu_cmd_set_common_fields(
        &vpu_load_probe_cmd, 1U << PROFILE_TILE_SRC_SLOT, 0U, 1U, 0U,
        PROFILE_TILE_ELEM_COUNT, sizeof(uint32_t), 0U, VPU_DATA_F32, 0U,
        0x60000000U + (PROFILE_TILE_SRC_SLOT * VPU_LOCAL_SLOT_STRIDE), 0U, 0U,
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER));
    vpu_cmd_init_raw(&vpu_compute_probe_cmd, PROFILE_TILE_VPU0_ID,
                     VPU_OP_VSCALE, 0U);
    vpu_cmd_set_common_fields(
        &vpu_compute_probe_cmd, 0x1U, 0x2U, 1U, 0U, PROFILE_TILE_ELEM_COUNT,
        sizeof(uint32_t), sizeof(uint32_t), VPU_DATA_F32,
        float_to_bits(0.5f),
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT_BUFFER),
        vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER));
    vpu_cmd_init_raw(&vpu_store_probe_cmd, PROFILE_TILE_VPU0_ID,
                     VPU_OP_VSTORE, PROFILE_TILE_VPU0_SYNC);
    vpu_cmd_set_common_fields(
        &vpu_store_probe_cmd, 0U, 1U << PROFILE_TILE_LINEAR_SLOT, 1U, 0U,
        PROFILE_TILE_ELEM_COUNT, 0U, sizeof(uint32_t), VPU_DATA_F32, 0U,
        vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER),
        0U, 0U,
        0x60000000U + (PROFILE_TILE_LINEAR_SLOT * VPU_LOCAL_SLOT_STRIDE));
    printf("PROFILE_TILE_RDCYCLE_OVERHEAD=%llu\n",
           (unsigned long long)g_cycle_probe_overhead);
    printf("PROFILE_TILE_BUILD_PROBE type=dma total=%llu\n",
           (unsigned long long)dma_build.total_cycles);
    printf("PROFILE_TILE_BUILD_PROBE type=sync total=%llu\n",
           (unsigned long long)sync_build.total_cycles);
    printf("PROFILE_TILE_BUILD_PROBE type=vpu_load init=%llu fill=%llu total=%llu\n",
           (unsigned long long)vpu_load_build.init_cycles,
           (unsigned long long)vpu_load_build.fill_cycles,
           (unsigned long long)vpu_load_build.total_cycles);
    printf("PROFILE_TILE_BUILD_PROBE type=vpu_compute init=%llu fill=%llu total=%llu\n",
           (unsigned long long)vpu_compute_build.init_cycles,
           (unsigned long long)vpu_compute_build.fill_cycles,
           (unsigned long long)vpu_compute_build.total_cycles);
    printf("PROFILE_TILE_BUILD_PROBE type=vpu_store init=%llu fill=%llu total=%llu\n",
           (unsigned long long)vpu_store_build.init_cycles,
           (unsigned long long)vpu_store_build.fill_cycles,
           (unsigned long long)vpu_store_build.total_cycles);
    printf("PROFILE_TILE_STAGE_PROBE type=dma cycles=%llu\n",
           (unsigned long long)measure_stage_cycles(&dma_probe_cmd));
    printf("PROFILE_TILE_STAGE_PROBE type=sync cycles=%llu\n",
           (unsigned long long)measure_stage_cycles(&sync_probe_cmd));
    printf("PROFILE_TILE_STAGE_PROBE type=vpu_load cycles=%llu\n",
           (unsigned long long)measure_stage_cycles(&vpu_load_probe_cmd));
    printf("PROFILE_TILE_STAGE_PROBE type=vpu_compute cycles=%llu\n",
           (unsigned long long)measure_stage_cycles(&vpu_compute_probe_cmd));
    printf("PROFILE_TILE_STAGE_PROBE type=vpu_store cycles=%llu\n",
           (unsigned long long)measure_stage_cycles(&vpu_store_probe_cmd));
#endif

    issue_profiled_dma_move(PROFILE_TILE_DMA_SYNC, PROFILE_TILE_DRAM_SRC_BASE,
                            PROFILE_TILE_SPM_BASE);
    issue_profiled_sync_wait(PROFILE_TILE_DMA_DEVICE_ID, PROFILE_TILE_DMA_SYNC);
    issue_profiled_vpu_load(PROFILE_TILE_VPU0_ID, PROFILE_TILE_SRC_SLOT,
                            PROFILE_TILE_ELEM_COUNT,
                            sizeof(uint32_t), VPU_DATA_F32);
    issue_profiled_vpu_compute(PROFILE_TILE_VPU0_ID, VPU_OP_VSCALE, 0x1U,
                               0x2U, 1U,
                               PROFILE_TILE_ELEM_COUNT, sizeof(uint32_t),
                               sizeof(uint32_t), VPU_DATA_F32,
                               float_to_bits(0.5f));
    issue_profiled_vpu_store(PROFILE_TILE_VPU0_ID, PROFILE_TILE_VPU0_SYNC,
                             PROFILE_TILE_LINEAR_SLOT,
                             PROFILE_TILE_ELEM_COUNT, sizeof(uint32_t),
                             VPU_DATA_F32);
    issue_profiled_sync_wait(PROFILE_TILE_VPU0_ID, PROFILE_TILE_VPU0_SYNC);
    issue_profiled_vpu_load(PROFILE_TILE_VPU1_ID, PROFILE_TILE_LINEAR_SLOT,
                            PROFILE_TILE_ELEM_COUNT,
                            sizeof(uint32_t), VPU_DATA_F32);
    issue_profiled_vpu_compute(PROFILE_TILE_VPU1_ID, VPU_OP_VSOFTMAX, 0x2U,
                               0x4U, 1U,
                               PROFILE_TILE_ELEM_COUNT, sizeof(uint32_t),
                               sizeof(uint32_t), VPU_DATA_F32, 0U);
    issue_profiled_vpu_store(PROFILE_TILE_VPU1_ID, PROFILE_TILE_VPU1_SYNC,
                             PROFILE_TILE_SOFTMAX_SLOT,
                             PROFILE_TILE_ELEM_COUNT, sizeof(uint32_t),
                             VPU_DATA_F32);
    issue_profiled_sync_wait(PROFILE_TILE_VPU1_ID, PROFILE_TILE_VPU1_SYNC);
    issue_profiled_dma_move(
        PROFILE_TILE_COPYBACK_SYNC,
        PROFILE_TILE_SPM_BASE +
            (PROFILE_TILE_SOFTMAX_SLOT *
             PROFILE_TILE_SLOT_STRIDE_BYTES),
        PROFILE_TILE_DRAM_DST_BASE);
    issue_profiled_sync_wait(PROFILE_TILE_DMA_DEVICE_ID,
                             PROFILE_TILE_COPYBACK_SYNC);

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
