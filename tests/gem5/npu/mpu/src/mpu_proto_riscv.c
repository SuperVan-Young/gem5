/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmd/common.hh"
#include "cmd/mpu.hh"
#include "npu_sync.hh"

#define MPU_DEVICE_ID 0x0U
#define DMA_DEVICE_ID 0x0U

#define DMA_MODE_MOVE_LAYOUT 0x0U
#define DMA_MEM_SPACE_DRAM 0x0U
#define DMA_MEM_SPACE_SPM 0x1U

#define DRAM_BASE 0x20000000UL
#define SPM_BASE 0x60000000UL

#define LOAD_A_SRC (SPM_BASE + 0x1000UL)
#define LOAD_B_SRC (SPM_BASE + 0x1100UL)
#define LOAD_C_SRC (SPM_BASE + 0x1200UL)
#define STORE_C_DST (SPM_BASE + 0x1300UL)

#define OFFSET_A_SRC (SPM_BASE + 0x1400UL)
#define OFFSET_B_SRC (SPM_BASE + 0x1500UL)
#define OFFSET_C_SRC (SPM_BASE + 0x1600UL)
#define OFFSET_C_DST (SPM_BASE + 0x1700UL)

#define TENSOR_A_BASE (SPM_BASE + 0x2000UL)
#define TENSOR_B_BASE (SPM_BASE + 0x2200UL)
#define TENSOR_C_BASE (SPM_BASE + 0x2400UL)

#define LOOP_K_A_BASE (SPM_BASE + 0x2800UL)
#define LOOP_K_B_BASE (SPM_BASE + 0x2C00UL)
#define LOOP_K_C_BASE (SPM_BASE + 0x3000UL)
#define PP_C_BASE (SPM_BASE + 0x3400UL)
#define K_OUTER_A_BASE (SPM_BASE + 0x3800UL)
#define K_OUTER_B_BASE (SPM_BASE + 0x3C00UL)
#define K_OUTER_C_BASE (SPM_BASE + 0x4200UL)
#define SKEW_C_BASE (SPM_BASE + 0x4600UL)

#define DRAM_A_SRC (DRAM_BASE + 0x1000UL)
#define DRAM_B_SRC (DRAM_BASE + 0x1100UL)
#define DRAM_C_DST (DRAM_BASE + 0x1200UL)

static const uint32_t kValidM = 2U;
static const uint32_t kValidN = 2U;
static const uint32_t kValidK = 3U;
static const size_t kATileBytes = 6U;
static const size_t kBTileBytes = 6U;
static const size_t kCTileBytes = 16U;
static const size_t kCTileElems = 4U;

static const int8_t kATile[] = {1, 2, 3, 4, 5, 6};
static const int8_t kBTile[] = {7, 8, 9, 10, 11, 12};
static const int32_t kColdTile[] = {5, 6, 7, 8};

static const int8_t kLoopA0[] = {1, 0, 2, 1, 1, 0};
static const int8_t kLoopA1[] = {0, 1, 1, 2, 1, 1};
static const int8_t kLoopB0[] = {1, 2, 0, 1, 1, 0};
static const int8_t kLoopB1[] = {2, 1, 1, 0, 0, 1};

static const int8_t kOuterA0[] = {1, 0, 1, 2, 0, 1};
static const int8_t kOuterA1[] = {0, 1, 2, 0, 1, 1};
static const int8_t kOuterB00[] = {1, 0, 1, 1, 0, 1};
static const int8_t kOuterB01[] = {2, 1, 0, 1, 1, 0};
static const int8_t kOuterB10[] = {0, 1, 1, 0, 1, 2};
static const int8_t kOuterB11[] = {1, 2, 1, 0, 0, 1};

typedef struct
{
    uint32_t h;
    uint32_t w;
    uint32_t c;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t stride_c;
    uint16_t k;
} DmaLayout;

static size_t
tile_physical_bytes(uint32_t rows, uint32_t cols, size_t elem_bytes,
                    uint32_t layout)
{
    size_t row_stride = (size_t)cols * elem_bytes;
    if (layout == MPU_LAYOUT_MODE_SKEWED && rows > 1U) {
        row_stride += elem_bytes;
    }
    return rows == 0U ? 0U : ((rows - 1U) * row_stride) + ((size_t)cols * elem_bytes);
}

static void
clear_bytes(uintptr_t base, size_t bytes)
{
    for (size_t i = 0; i < bytes; ++i) {
        *(volatile uint8_t *)(base + i) = 0;
    }
}

static void
write_external_i8_tile(uintptr_t base, const int8_t *values, uint32_t rows,
                       uint32_t cols, uint32_t layout)
{
    const size_t elem_bytes = 1U;
    const size_t row_stride =
        ((size_t)cols * elem_bytes) +
        ((layout == MPU_LAYOUT_MODE_SKEWED && rows > 1U) ? elem_bytes : 0U);
    size_t linear_index = 0U;

    clear_bytes(base, tile_physical_bytes(rows, cols, elem_bytes, layout));
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            *(volatile int8_t *)(base + row * row_stride + col) =
                values[linear_index++];
        }
    }
}

static void
write_external_i32_tile(uintptr_t base, const int32_t *values, uint32_t rows,
                        uint32_t cols, uint32_t layout)
{
    const size_t elem_bytes = sizeof(int32_t);
    const size_t row_stride =
        ((size_t)cols * elem_bytes) +
        ((layout == MPU_LAYOUT_MODE_SKEWED && rows > 1U) ? elem_bytes : 0U);
    size_t linear_index = 0U;

    clear_bytes(base, tile_physical_bytes(rows, cols, elem_bytes, layout));
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            *(volatile int32_t *)(base + row * row_stride + col * elem_bytes) =
                values[linear_index++];
        }
    }
}

static int
verify_external_i32_tile(uintptr_t base, const int32_t *values, uint32_t rows,
                         uint32_t cols, uint32_t layout)
{
    const size_t elem_bytes = sizeof(int32_t);
    const size_t row_stride =
        ((size_t)cols * elem_bytes) +
        ((layout == MPU_LAYOUT_MODE_SKEWED && rows > 1U) ? elem_bytes : 0U);
    size_t linear_index = 0U;

    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const int32_t observed =
                *(volatile int32_t *)(base + row * row_stride + col * elem_bytes);
            if (observed != values[linear_index++]) {
                return 0;
            }
        }
    }

    return 1;
}

static void
reference_matmul(const int8_t *a, const int8_t *b, const int32_t *c_old,
                 int32_t *out, uint32_t valid_m, uint32_t valid_n,
                 uint32_t valid_k, int accumulate)
{
    for (uint32_t m = 0; m < valid_m; ++m) {
        for (uint32_t n = 0; n < valid_n; ++n) {
            int32_t value = accumulate ? c_old[m * valid_n + n] : 0;
            for (uint32_t k = 0; k < valid_k; ++k) {
                value += (int32_t)a[m * valid_k + k] *
                         (int32_t)b[k * valid_n + n];
            }
            out[m * valid_n + n] = value;
        }
    }
}

static void
reference_two_stage_acc(const int8_t *a0, const int8_t *b0, const int8_t *a1,
                        const int8_t *b1, int32_t *out)
{
    int32_t tmp[kCTileElems];

    reference_matmul(a0, b0, NULL, tmp, kValidM, kValidN, kValidK, 0);
    reference_matmul(a1, b1, tmp, out, kValidM, kValidN, kValidK, 1);
}

static void
reference_three_stage_acc(const int8_t *a0, const int8_t *b0,
                          const int8_t *a1, const int8_t *b1,
                          const int32_t *c0_old, int32_t *out,
                          int use_old_c)
{
    int32_t tmp[kCTileElems];

    reference_matmul(a0, b0, use_old_c ? c0_old : NULL, tmp,
                     kValidM, kValidN, kValidK, use_old_c);
    reference_matmul(a1, b1, tmp, out, kValidM, kValidN, kValidK, 1);
}

static inline void
launch_mpu_load(uint32_t dst_local_addr, uintptr_t src_spm_addr,
                uint32_t valid_m, uint32_t valid_n, uint32_t valid_k,
                uint32_t layout_mode, uint32_t local_offset_bytes,
                uint32_t sync_indicator, uint32_t set_completion_sync)
{
    NpuCmd cmd;

    npuBuildMpuLoadCmd(&cmd, MPU_DEVICE_ID, dst_local_addr,
                       (uint32_t)src_spm_addr, valid_m, valid_n, valid_k,
                       layout_mode, local_offset_bytes, sync_indicator,
                       set_completion_sync);
    cmd.launchCmd();
}

static inline void
launch_mpu_compute(uint32_t src_local_addr_a, uint32_t src_local_addr_b,
                   uint32_t dst_local_addr_c, uint32_t valid_m,
                   uint32_t valid_n, uint32_t valid_k, uint32_t subop,
                   uint32_t dst_layout_mode, uint32_t local_offset_a,
                   uint32_t local_offset_b, uint32_t local_offset_c,
                   uint32_t sync_indicator, uint32_t set_completion_sync)
{
    NpuCmd cmd;

    npuBuildMpuComputeCmd(&cmd, MPU_DEVICE_ID, src_local_addr_a,
                          src_local_addr_b, dst_local_addr_c, valid_m,
                          valid_n, valid_k, subop, dst_layout_mode, 0U,
                          local_offset_a, local_offset_b, local_offset_c,
                          sync_indicator, set_completion_sync);
    cmd.launchCmd();
}

static inline void
launch_mpu_store(uint32_t src_local_addr, uintptr_t dst_spm_addr,
                 uint32_t valid_m, uint32_t valid_n, uint32_t valid_k,
                 uint32_t layout_mode, uint32_t local_offset_bytes,
                 uint32_t sync_indicator, uint32_t set_completion_sync)
{
    NpuCmd cmd;

    npuBuildMpuStoreCmd(&cmd, MPU_DEVICE_ID, src_local_addr,
                        (uint32_t)dst_spm_addr, valid_m, valid_n, valid_k,
                        layout_mode, local_offset_bytes, sync_indicator,
                        set_completion_sync);
    cmd.launchCmd();
}

static inline void
launch_mpu_tensor_loop(uint32_t base_local_addr_a,
                       uint32_t base_local_addr_b,
                       uint32_t base_local_addr_c,
                       uintptr_t base_spm_addr_a,
                       uintptr_t base_spm_addr_b,
                       uintptr_t base_spm_addr_c, uint32_t valid_m,
                       uint32_t valid_n, uint32_t valid_k, uint32_t subop,
                       uint32_t layout_a, uint32_t layout_b,
                       uint32_t layout_c, uint32_t outer_axis,
                       uint32_t outer_count, uint32_t inner_axis,
                       uint32_t inner_count, uint32_t offset_a,
                       uint32_t offset_b, uint32_t offset_c,
                       uint32_t outer_step_tiles,
                       uint32_t inner_step_tiles,
                       uint32_t pingpong_a_enable,
                       uint32_t pingpong_b_enable,
                       uint32_t pingpong_c_enable,
                       uint32_t auto_load_c_for_acc,
                       uint32_t sync_indicator,
                       uint32_t set_completion_sync)
{
    NpuCmd cmd;
    const uint32_t loop_ctrl0 = mpuBuildLoopCtrl0(
        subop, layout_a, layout_b, layout_c, outer_axis, inner_axis);
    const uint32_t offset_pack = mpuBuildOffsetPack(
        offset_a, offset_b, offset_c);
    const uint32_t step_cfg = mpuBuildStepCfg(
        outer_step_tiles, inner_step_tiles, pingpong_a_enable,
        pingpong_b_enable, pingpong_c_enable, auto_load_c_for_acc);

    npuBuildMpuTensorLoopCmd(&cmd, MPU_DEVICE_ID, base_local_addr_a,
                             base_local_addr_b, base_local_addr_c,
                             (uint32_t)base_spm_addr_a,
                             (uint32_t)base_spm_addr_b,
                             (uint32_t)base_spm_addr_c, valid_m, valid_n,
                             valid_k, loop_ctrl0, outer_count, inner_count,
                             offset_pack, step_cfg, sync_indicator,
                             set_completion_sync);
    cmd.launchCmd();
}

static inline DmaLayout
make_linear_layout(uint32_t bytes)
{
    DmaLayout layout = {1U, 1U, bytes, bytes, bytes, 1U, 0U};
    return layout;
}

static inline uint32_t
dma_mem_space_for_base(uintptr_t base)
{
    return base >= SPM_BASE ? DMA_MEM_SPACE_SPM : DMA_MEM_SPACE_DRAM;
}

static inline void
build_dma_move_layout_cmd(NpuCmd *cmd, uintptr_t src_base, uintptr_t dst_base,
                          DmaLayout src_layout, DmaLayout dst_layout,
                          uint32_t sync_indicator,
                          uint32_t set_completion_sync)
{
    const uint32_t op_code =
        ((0U & 0x7U) << 5) | ((DMA_MODE_MOVE_LAYOUT & 0x7U) << 2);
    const uint32_t mode_cfg =
        (dma_mem_space_for_base(src_base) & 0x1U) |
        ((dma_mem_space_for_base(dst_base) & 0x1U) << 1);

    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_DMA);
    cmd->setDeviceId(DMA_DEVICE_ID);
    cmd->setOpCode(op_code);
    cmd->setSyncIndicator(sync_indicator);
    cmd->setSetIndicatorSns(set_completion_sync ? 1U : 0U);
    cmd->setSetIndicatorSnd(0U);
    cmd->clearCommonReservedBits();
    cmd->setWord(1U, (uint32_t)src_base);
    cmd->setWord(2U, (uint32_t)dst_base);
    cmd->setWord(3U, src_layout.h);
    cmd->setWord(4U, src_layout.w);
    cmd->setWord(5U, src_layout.c);
    cmd->setWord(6U, src_layout.stride_h);
    cmd->setWord(7U, src_layout.stride_w);
    cmd->setWord(8U, src_layout.stride_c);
    cmd->setWord(9U, dst_layout.stride_h);
    cmd->setWord(10U, dst_layout.stride_w);
    cmd->setWord(11U, dst_layout.stride_c);
    cmd->setWord(12U, ((uint32_t)dst_layout.k << 16) | src_layout.k);
    cmd->setWord(13U, mode_cfg);
    cmd->setWord(14U, 0U);
    cmd->setWord(15U, 0U);
}

static inline void
launch_dma_move_layout(uintptr_t src_base, uintptr_t dst_base, uint32_t bytes,
                       uint32_t sync_indicator,
                       uint32_t set_completion_sync)
{
    NpuCmd cmd;
    const DmaLayout src_layout = make_linear_layout(bytes);
    const DmaLayout dst_layout = make_linear_layout(bytes);

    build_dma_move_layout_cmd(
        &cmd, src_base, dst_base, src_layout, dst_layout, sync_indicator,
        set_completion_sync);
    cmd.launchCmd();
}

static int
scenario_load_a(void)
{
    write_external_i8_tile(LOAD_A_SRC, kATile, kValidM, kValidK,
                           MPU_LAYOUT_MODE_NORMAL);
    launch_mpu_load(MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 1U, 0U);
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_load_c(void)
{
    write_external_i32_tile(LOAD_C_SRC, kColdTile, kValidM, kValidN,
                            MPU_LAYOUT_MODE_NORMAL);
    launch_mpu_load(MPU_LOCAL_ADDR_C0, LOAD_C_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 2U, 0U);
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_matmul_basic(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(STORE_C_DST, tile_physical_bytes(kValidM, kValidN, 4U,
                                                 MPU_LAYOUT_MODE_NORMAL));
    write_external_i8_tile(LOAD_A_SRC, kATile, kValidM, kValidK,
                           MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOAD_B_SRC, kBTile, kValidK, kValidN,
                           MPU_LAYOUT_MODE_NORMAL);
    reference_matmul(kATile, kBTile, NULL, expected, kValidM, kValidN,
                     kValidK, 0);

    launch_mpu_load(MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 3U, 0U);
    launch_mpu_load(MPU_LOCAL_ADDR_B0, LOAD_B_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 4U, 0U);
    launch_mpu_compute(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                       MPU_LOCAL_ADDR_C0, kValidM, kValidN, kValidK,
                       MPU_SUBOP_MATMUL, MPU_LAYOUT_MODE_NORMAL,
                       0U, 0U, 0U, 5U, 0U);
    launch_mpu_store(MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN,
                     kValidK, MPU_LAYOUT_MODE_NORMAL, 0U, 6U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(STORE_C_DST, expected, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_matmul_acc_basic(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(STORE_C_DST, tile_physical_bytes(kValidM, kValidN, 4U,
                                                 MPU_LAYOUT_MODE_NORMAL));
    write_external_i8_tile(LOAD_A_SRC, kATile, kValidM, kValidK,
                           MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOAD_B_SRC, kBTile, kValidK, kValidN,
                           MPU_LAYOUT_MODE_NORMAL);
    write_external_i32_tile(LOAD_C_SRC, kColdTile, kValidM, kValidN,
                            MPU_LAYOUT_MODE_NORMAL);
    reference_matmul(kATile, kBTile, kColdTile, expected, kValidM, kValidN,
                     kValidK, 1);

    launch_mpu_load(MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 7U, 0U);
    launch_mpu_load(MPU_LOCAL_ADDR_B0, LOAD_B_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 8U, 0U);
    launch_mpu_load(MPU_LOCAL_ADDR_C0, LOAD_C_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 9U, 0U);
    launch_mpu_compute(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                       MPU_LOCAL_ADDR_C0, kValidM, kValidN, kValidK,
                       MPU_SUBOP_MATMUL_ACC, MPU_LAYOUT_MODE_NORMAL,
                       0U, 0U, 0U, 10U, 0U);
    launch_mpu_store(MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN,
                     kValidK, MPU_LAYOUT_MODE_NORMAL, 0U, 11U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(STORE_C_DST, expected, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_sync_completion(void)
{
    clear_bytes(STORE_C_DST, tile_physical_bytes(kValidM, kValidN, 4U,
                                                 MPU_LAYOUT_MODE_NORMAL));
    write_external_i32_tile(LOAD_C_SRC, kColdTile, kValidM, kValidN,
                            MPU_LAYOUT_MODE_NORMAL);

    launch_mpu_load(MPU_LOCAL_ADDR_C0, LOAD_C_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 12U, 1U);
    npu_launch_sync_wait(MPU_DEVICE_ID, 12U, 0U, 0U, 0U);
    launch_mpu_store(MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN,
                     kValidK, MPU_LAYOUT_MODE_NORMAL, 0U, 13U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(STORE_C_DST, kColdTile, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_tensor_loop_k_inner_local_accumulate(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(LOOP_K_C_BASE, tile_physical_bytes(kValidM, kValidN, 4U,
                                                   MPU_LAYOUT_MODE_NORMAL));
    write_external_i8_tile(LOOP_K_A_BASE + 0U * kATileBytes, kLoopA0,
                           kValidM, kValidK, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOOP_K_A_BASE + 1U * kATileBytes, kLoopA1,
                           kValidM, kValidK, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOOP_K_B_BASE + 0U * kBTileBytes, kLoopB0,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOOP_K_B_BASE + 1U * kBTileBytes, kLoopB1,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    reference_two_stage_acc(kLoopA0, kLoopB0, kLoopA1, kLoopB1, expected);

    launch_mpu_tensor_loop(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                           MPU_LOCAL_ADDR_C0, LOOP_K_A_BASE, LOOP_K_B_BASE,
                           LOOP_K_C_BASE, kValidM, kValidN, kValidK,
                           MPU_SUBOP_MATMUL, MPU_LAYOUT_MODE_NORMAL,
                           MPU_LAYOUT_MODE_NORMAL, MPU_LAYOUT_MODE_NORMAL,
                           MPU_AXIS_N, 1U, MPU_AXIS_K, 2U,
                           0U, 0U, 0U, 1U, 1U, 0U, 0U, 0U, 0U, 14U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(LOOP_K_C_BASE, expected, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_tensor_loop_k_outer_spill_reload(void)
{
    int32_t expected0[kCTileElems];
    int32_t expected1[kCTileElems];

    clear_bytes(K_OUTER_C_BASE, 2U * tile_physical_bytes(kValidM, kValidN, 4U,
                                                         MPU_LAYOUT_MODE_NORMAL));
    write_external_i8_tile(K_OUTER_A_BASE + 0U * kATileBytes, kOuterA0,
                           kValidM, kValidK, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_A_BASE + 1U * kATileBytes, kOuterA1,
                           kValidM, kValidK, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_B_BASE + 0U * kBTileBytes, kOuterB00,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_B_BASE + 1U * kBTileBytes, kOuterB01,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_B_BASE + 2U * kBTileBytes, kOuterB10,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_B_BASE + 3U * kBTileBytes, kOuterB11,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    reference_two_stage_acc(kOuterA0, kOuterB00, kOuterA1, kOuterB10, expected0);
    reference_two_stage_acc(kOuterA0, kOuterB01, kOuterA1, kOuterB11, expected1);

    launch_mpu_tensor_loop(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                           MPU_LOCAL_ADDR_C0, K_OUTER_A_BASE, K_OUTER_B_BASE,
                           K_OUTER_C_BASE, kValidM, kValidN, kValidK,
                           MPU_SUBOP_MATMUL, MPU_LAYOUT_MODE_NORMAL,
                           MPU_LAYOUT_MODE_NORMAL, MPU_LAYOUT_MODE_NORMAL,
                           MPU_AXIS_K, 2U, MPU_AXIS_N, 2U,
                           0U, 0U, 0U, 1U, 1U, 0U, 0U, 0U, 0U, 15U, 0U);
    npu_cmd_sync_done();

    if (!verify_external_i32_tile(K_OUTER_C_BASE + 0U * kCTileBytes, expected0,
                                  kValidM, kValidN, MPU_LAYOUT_MODE_NORMAL)) {
        return 1;
    }
    return verify_external_i32_tile(K_OUTER_C_BASE + 1U * kCTileBytes,
                                    expected1, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_tensor_loop_k_outer_pingpong_c(void)
{
    int32_t expected0[kCTileElems];
    int32_t expected1[kCTileElems];

    clear_bytes(PP_C_BASE, 2U * tile_physical_bytes(kValidM, kValidN, 4U,
                                                    MPU_LAYOUT_MODE_NORMAL));
    write_external_i8_tile(K_OUTER_A_BASE + 0U * kATileBytes, kOuterA0,
                           kValidM, kValidK, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_A_BASE + 1U * kATileBytes, kOuterA1,
                           kValidM, kValidK, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_B_BASE + 0U * kBTileBytes, kOuterB00,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_B_BASE + 1U * kBTileBytes, kOuterB01,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_B_BASE + 2U * kBTileBytes, kOuterB10,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(K_OUTER_B_BASE + 3U * kBTileBytes, kOuterB11,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    reference_two_stage_acc(kOuterA0, kOuterB00, kOuterA1, kOuterB10, expected0);
    reference_two_stage_acc(kOuterA0, kOuterB01, kOuterA1, kOuterB11, expected1);

    launch_mpu_tensor_loop(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                           MPU_LOCAL_ADDR_C0, K_OUTER_A_BASE, K_OUTER_B_BASE,
                           PP_C_BASE, kValidM, kValidN, kValidK,
                           MPU_SUBOP_MATMUL, MPU_LAYOUT_MODE_NORMAL,
                           MPU_LAYOUT_MODE_NORMAL, MPU_LAYOUT_MODE_NORMAL,
                           MPU_AXIS_K, 2U, MPU_AXIS_N, 2U,
                           0U, 0U, 0U, 1U, 1U, 0U, 0U, 1U, 0U, 16U, 0U);
    npu_cmd_sync_done();

    if (!verify_external_i32_tile(PP_C_BASE + 0U * kCTileBytes, expected0,
                                  kValidM, kValidN, MPU_LAYOUT_MODE_NORMAL)) {
        return 1;
    }
    return verify_external_i32_tile(PP_C_BASE + 1U * kCTileBytes, expected1,
                                    kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_tensor_loop_matmul_acc_first_k_legacy_overwrite(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(LOOP_K_C_BASE, tile_physical_bytes(kValidM, kValidN, 4U,
                                                   MPU_LAYOUT_MODE_NORMAL));
    write_external_i8_tile(LOOP_K_A_BASE + 0U * kATileBytes, kLoopA0,
                           kValidM, kValidK, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOOP_K_A_BASE + 1U * kATileBytes, kLoopA1,
                           kValidM, kValidK, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOOP_K_B_BASE + 0U * kBTileBytes, kLoopB0,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOOP_K_B_BASE + 1U * kBTileBytes, kLoopB1,
                           kValidK, kValidN, MPU_LAYOUT_MODE_NORMAL);
    write_external_i32_tile(LOOP_K_C_BASE, kColdTile, kValidM, kValidN,
                            MPU_LAYOUT_MODE_NORMAL);
    reference_two_stage_acc(kLoopA0, kLoopB0, kLoopA1, kLoopB1, expected);

    launch_mpu_tensor_loop(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                           MPU_LOCAL_ADDR_C0, LOOP_K_A_BASE, LOOP_K_B_BASE,
                           LOOP_K_C_BASE, kValidM, kValidN, kValidK,
                           MPU_SUBOP_MATMUL_ACC, MPU_LAYOUT_MODE_NORMAL,
                           MPU_LAYOUT_MODE_NORMAL, MPU_LAYOUT_MODE_NORMAL,
                           MPU_AXIS_N, 1U, MPU_AXIS_K, 2U,
                           0U, 0U, 0U, 1U, 1U, 0U, 0U, 0U, 1U, 17U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(LOOP_K_C_BASE, expected, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_explicit_offset_load_store(void)
{
    clear_bytes(OFFSET_C_DST, tile_physical_bytes(kValidM, kValidN, 4U,
                                                  MPU_LAYOUT_MODE_NORMAL));
    write_external_i32_tile(OFFSET_C_SRC, kColdTile, kValidM, kValidN,
                            MPU_LAYOUT_MODE_NORMAL);

    launch_mpu_load(MPU_LOCAL_ADDR_C0, OFFSET_C_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 4U, 18U, 0U);
    launch_mpu_store(MPU_LOCAL_ADDR_C0, OFFSET_C_DST, kValidM, kValidN, kValidK,
                     MPU_LAYOUT_MODE_NORMAL, 4U, 19U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(OFFSET_C_DST, kColdTile, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_explicit_offset_compute(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(OFFSET_C_DST, tile_physical_bytes(kValidM, kValidN, 4U,
                                                  MPU_LAYOUT_MODE_NORMAL));
    write_external_i8_tile(OFFSET_A_SRC, kATile, kValidM, kValidK,
                           MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(OFFSET_B_SRC, kBTile, kValidK, kValidN,
                           MPU_LAYOUT_MODE_NORMAL);
    reference_matmul(kATile, kBTile, NULL, expected, kValidM, kValidN,
                     kValidK, 0);

    launch_mpu_load(MPU_LOCAL_ADDR_A0, OFFSET_A_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 1U, 20U, 0U);
    launch_mpu_load(MPU_LOCAL_ADDR_B0, OFFSET_B_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 2U, 21U, 0U);
    launch_mpu_compute(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                       MPU_LOCAL_ADDR_C0, kValidM, kValidN, kValidK,
                       MPU_SUBOP_MATMUL, MPU_LAYOUT_MODE_NORMAL,
                       1U, 2U, 4U, 22U, 0U);
    launch_mpu_store(MPU_LOCAL_ADDR_C0, OFFSET_C_DST, kValidM, kValidN,
                     kValidK, MPU_LAYOUT_MODE_NORMAL, 4U, 23U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(OFFSET_C_DST, expected, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_tensor_loop_layout_fields_outside_step_cfg(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(SKEW_C_BASE, tile_physical_bytes(kValidM, kValidN, 4U,
                                                 MPU_LAYOUT_MODE_SKEWED));
    write_external_i8_tile(TENSOR_A_BASE, kATile, kValidM, kValidK,
                           MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(TENSOR_B_BASE, kBTile, kValidK, kValidN,
                           MPU_LAYOUT_MODE_NORMAL);
    reference_matmul(kATile, kBTile, NULL, expected, kValidM, kValidN,
                     kValidK, 0);

    launch_mpu_tensor_loop(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                           MPU_LOCAL_ADDR_C0, TENSOR_A_BASE, TENSOR_B_BASE,
                           SKEW_C_BASE, kValidM, kValidN, kValidK,
                           MPU_SUBOP_MATMUL, MPU_LAYOUT_MODE_NORMAL,
                           MPU_LAYOUT_MODE_NORMAL, MPU_LAYOUT_MODE_SKEWED,
                           MPU_AXIS_M, 1U, MPU_AXIS_N, 1U,
                           0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 24U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(SKEW_C_BASE, expected, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_SKEWED)
               ? 0
               : 1;
}

static int
scenario_store_layout_conversion_normal_to_skew(void)
{
    clear_bytes(SKEW_C_BASE, tile_physical_bytes(kValidM, kValidN, 4U,
                                                 MPU_LAYOUT_MODE_SKEWED));
    write_external_i32_tile(LOAD_C_SRC, kColdTile, kValidM, kValidN,
                            MPU_LAYOUT_MODE_NORMAL);

    launch_mpu_load(MPU_LOCAL_ADDR_C0, LOAD_C_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 25U, 0U);
    launch_mpu_store(MPU_LOCAL_ADDR_C0, SKEW_C_BASE, kValidM, kValidN, kValidK,
                     MPU_LAYOUT_MODE_SKEWED, 0U, 26U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(SKEW_C_BASE, kColdTile, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_SKEWED)
               ? 0
               : 1;
}

static int
scenario_store_layout_conversion_skew_to_normal(void)
{
    clear_bytes(STORE_C_DST, tile_physical_bytes(kValidM, kValidN, 4U,
                                                 MPU_LAYOUT_MODE_NORMAL));
    write_external_i32_tile(SKEW_C_BASE, kColdTile, kValidM, kValidN,
                            MPU_LAYOUT_MODE_SKEWED);

    launch_mpu_load(MPU_LOCAL_ADDR_C0, SKEW_C_BASE, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_SKEWED, 0U, 27U, 0U);
    launch_mpu_store(MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN, kValidK,
                     MPU_LAYOUT_MODE_NORMAL, 0U, 28U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(STORE_C_DST, kColdTile, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_observed_spm_stall_from_retry(void)
{
    return scenario_tensor_loop_matmul_acc_first_k_legacy_overwrite();
}

static int
scenario_observed_slot_stall_from_bank_conflict(void)
{
    return scenario_explicit_offset_compute();
}

static int
scenario_multi_mem_port_tensor_loop_throughput(void)
{
    return scenario_tensor_loop_matmul_acc_first_k_legacy_overwrite();
}

static int
scenario_dma_to_mpu_to_dma_regression(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(STORE_C_DST, kCTileBytes);
    clear_bytes(DRAM_C_DST, kCTileBytes);
    write_external_i8_tile(DRAM_A_SRC, kATile, kValidM, kValidK,
                           MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(DRAM_B_SRC, kBTile, kValidK, kValidN,
                           MPU_LAYOUT_MODE_NORMAL);
    reference_matmul(kATile, kBTile, NULL, expected, kValidM, kValidN,
                     kValidK, 0);

    launch_dma_move_layout(DRAM_A_SRC, LOAD_A_SRC, (uint32_t)kATileBytes, 41U, 1U);
    launch_dma_move_layout(DRAM_B_SRC, LOAD_B_SRC, (uint32_t)kBTileBytes, 42U, 1U);
    npu_launch_sync_wait(DMA_DEVICE_ID, 41U, 0U, 0U, 0U);
    npu_launch_sync_wait(DMA_DEVICE_ID, 42U, 0U, 0U, 0U);
    launch_mpu_load(MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 43U, 0U);
    launch_mpu_load(MPU_LOCAL_ADDR_B0, LOAD_B_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 44U, 0U);
    launch_mpu_compute(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                       MPU_LOCAL_ADDR_C0, kValidM, kValidN, kValidK,
                       MPU_SUBOP_MATMUL, MPU_LAYOUT_MODE_NORMAL,
                       0U, 0U, 0U, 45U, 0U);
    launch_mpu_store(MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN,
                     kValidK, MPU_LAYOUT_MODE_NORMAL, 0U, 46U, 1U);
    npu_launch_sync_wait(MPU_DEVICE_ID, 46U, 0U, 0U, 0U);
    launch_dma_move_layout(STORE_C_DST, DRAM_C_DST, (uint32_t)kCTileBytes, 47U, 0U);
    npu_cmd_sync_done();

    return verify_external_i32_tile(DRAM_C_DST, expected, kValidM, kValidN,
                                    MPU_LAYOUT_MODE_NORMAL)
               ? 0
               : 1;
}

static int
scenario_low_bits_in_local_addr_rejected(void)
{
    NpuCmd cmd;

    write_external_i8_tile(LOAD_A_SRC, kATile, kValidM, kValidK,
                           MPU_LAYOUT_MODE_NORMAL);
    npuBuildMpuLoadCmd(&cmd, MPU_DEVICE_ID, MPU_LOCAL_ADDR_A0 + 1U,
                       (uint32_t)LOAD_A_SRC, kValidM, kValidN, kValidK,
                       MPU_LAYOUT_MODE_NORMAL, 0U, 60U, 0U);
    cmd.launchCmd();
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_matmul_acc_missing_c(void)
{
    write_external_i8_tile(LOAD_A_SRC, kATile, kValidM, kValidK,
                           MPU_LAYOUT_MODE_NORMAL);
    write_external_i8_tile(LOAD_B_SRC, kBTile, kValidK, kValidN,
                           MPU_LAYOUT_MODE_NORMAL);

    launch_mpu_load(MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 61U, 0U);
    launch_mpu_load(MPU_LOCAL_ADDR_B0, LOAD_B_SRC, kValidM, kValidN, kValidK,
                    MPU_LAYOUT_MODE_NORMAL, 0U, 62U, 0U);
    launch_mpu_compute(MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
                       MPU_LOCAL_ADDR_C0, kValidM, kValidN, kValidK,
                       MPU_SUBOP_MATMUL_ACC, MPU_LAYOUT_MODE_NORMAL,
                       0U, 0U, 0U, 63U, 0U);
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_tensor_loop_auto_load_c_for_acc_required(void)
{
    NpuCmd cmd;
    const uint32_t loop_ctrl0 = mpuBuildLoopCtrl0(
        MPU_SUBOP_MATMUL_ACC, MPU_LAYOUT_MODE_NORMAL, MPU_LAYOUT_MODE_NORMAL,
        MPU_LAYOUT_MODE_NORMAL, MPU_AXIS_N, MPU_AXIS_K);
    const uint32_t offset_pack = mpuBuildOffsetPack(0U, 0U, 0U);
    const uint32_t step_cfg = mpuBuildStepCfg(1U, 1U, 0U, 0U, 0U, 0U);

    npuBuildMpuTensorLoopCmd(&cmd, MPU_DEVICE_ID, MPU_LOCAL_ADDR_A0,
                             MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0,
                             (uint32_t)LOOP_K_A_BASE, (uint32_t)LOOP_K_B_BASE,
                             (uint32_t)LOOP_K_C_BASE, kValidM, kValidN,
                             kValidK, loop_ctrl0, 1U, 2U, offset_pack,
                             step_cfg, 64U, 0U);
    cmd.launchCmd();
    npu_cmd_sync_done();
    return 0;
}

int
main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <scenario>\n", argv[0]);
        return 2;
    }

    if (strcmp(argv[1], "load_a") == 0) {
        return scenario_load_a();
    }
    if (strcmp(argv[1], "load_c") == 0) {
        return scenario_load_c();
    }
    if (strcmp(argv[1], "matmul_basic") == 0) {
        return scenario_matmul_basic();
    }
    if (strcmp(argv[1], "matmul_acc_basic") == 0) {
        return scenario_matmul_acc_basic();
    }
    if (strcmp(argv[1], "sync_completion") == 0) {
        return scenario_sync_completion();
    }
    if (strcmp(argv[1], "tensor_loop_k_inner_local_accumulate") == 0) {
        return scenario_tensor_loop_k_inner_local_accumulate();
    }
    if (strcmp(argv[1], "tensor_loop_k_outer_spill_reload") == 0) {
        return scenario_tensor_loop_k_outer_spill_reload();
    }
    if (strcmp(argv[1], "tensor_loop_k_outer_pingpong_c") == 0) {
        return scenario_tensor_loop_k_outer_pingpong_c();
    }
    if (strcmp(argv[1], "tensor_loop_matmul_acc_first_k_legacy_overwrite") == 0) {
        return scenario_tensor_loop_matmul_acc_first_k_legacy_overwrite();
    }
    if (strcmp(argv[1], "explicit_offset_load_store") == 0) {
        return scenario_explicit_offset_load_store();
    }
    if (strcmp(argv[1], "explicit_offset_compute") == 0) {
        return scenario_explicit_offset_compute();
    }
    if (strcmp(argv[1], "tensor_loop_layout_fields_outside_step_cfg") == 0) {
        return scenario_tensor_loop_layout_fields_outside_step_cfg();
    }
    if (strcmp(argv[1], "store_layout_conversion_normal_to_skew") == 0) {
        return scenario_store_layout_conversion_normal_to_skew();
    }
    if (strcmp(argv[1], "store_layout_conversion_skew_to_normal") == 0) {
        return scenario_store_layout_conversion_skew_to_normal();
    }
    if (strcmp(argv[1], "observed_spm_stall_from_retry") == 0) {
        return scenario_observed_spm_stall_from_retry();
    }
    if (strcmp(argv[1], "observed_slot_stall_from_bank_conflict") == 0) {
        return scenario_observed_slot_stall_from_bank_conflict();
    }
    if (strcmp(argv[1], "multi_mem_port_tensor_loop_throughput") == 0) {
        return scenario_multi_mem_port_tensor_loop_throughput();
    }
    if (strcmp(argv[1], "dma_to_mpu_to_dma_regression") == 0) {
        return scenario_dma_to_mpu_to_dma_regression();
    }
    if (strcmp(argv[1], "low_bits_in_local_addr_rejected") == 0) {
        return scenario_low_bits_in_local_addr_rejected();
    }
    if (strcmp(argv[1], "matmul_acc_missing_c") == 0) {
        return scenario_matmul_acc_missing_c();
    }
    if (strcmp(argv[1], "tensor_loop_auto_load_c_for_acc_required") == 0) {
        return scenario_tensor_loop_auto_load_c_for_acc_required();
    }

    fprintf(stderr, "unknown scenario: %s\n", argv[1]);
    return 2;
}
