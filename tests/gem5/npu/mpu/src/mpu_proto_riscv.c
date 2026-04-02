/*
 * Copyright (c) 2026
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
#define LOOP_K_B_BASE (SPM_BASE + 0x2c00UL)
#define LOOP_K_C_BASE (SPM_BASE + 0x3000UL)
#define LOOP_K_C_OLD_BASE (SPM_BASE + 0x3200UL)
#define PP_C_BASE (SPM_BASE + 0x3600UL)

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

static const int8_t kTensorATile0[] = {1, 2, 3, 4, 5, 6};
static const int8_t kTensorATile1[] = {-1, 1, 2, 0, 3, 4};
static const int8_t kTensorBTile0[] = {1, 0, 0, 1, 1, 1};
static const int8_t kTensorBTile1[] = {2, 1, 1, 0, 0, 1};

static const int8_t kLoopKATile0[] = {1, 0, 2, 1, 1, 0};
static const int8_t kLoopKATile1[] = {0, 1, 1, 2, 1, 1};
static const int8_t kLoopKBTile0[] = {1, 2, 0, 1, 1, 0};
static const int8_t kLoopKBTile1[] = {2, 1, 1, 0, 0, 1};

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

static inline uint32_t
compose_local_addr(uint32_t base, uint32_t offset)
{
    return base + offset;
}

static inline void
clear_bytes(uintptr_t base, size_t bytes)
{
    for (size_t i = 0; i < bytes; ++i) {
        *(volatile uint8_t *)(base + i) = 0;
    }
}

static void
write_i8_tile(uintptr_t base, const int8_t *values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        *(volatile int8_t *)(base + i) = values[i];
    }
}

static void
write_i32_tile(uintptr_t base, const int32_t *values, size_t count)
{
    volatile int32_t *ptr = (volatile int32_t *)base;

    for (size_t i = 0; i < count; ++i) {
        ptr[i] = values[i];
    }
}

static int
verify_i32_tile(uintptr_t base, const int32_t *values, size_t count)
{
    volatile int32_t *ptr = (volatile int32_t *)base;

    for (size_t i = 0; i < count; ++i) {
        if (ptr[i] != values[i]) {
            return 0;
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
reference_two_stage_acc(const int8_t *a0, const int8_t *b0,
                        const int8_t *a1, const int8_t *b1,
                        int32_t *out)
{
    int32_t tmp[kCTileElems];

    reference_matmul(a0, b0, NULL, tmp, kValidM, kValidN, kValidK, 0);
    reference_matmul(a1, b1, tmp, out, kValidM, kValidN, kValidK, 1);
}

static inline void
launch_mpu_load_ext(uint32_t dst_local_addr, uintptr_t src_spm_addr,
                    uint32_t valid_m, uint32_t valid_n, uint32_t valid_k,
                    uint32_t layout_mode, uint32_t sync_indicator,
                    uint32_t set_completion_sync)
{
    NpuCmd cmd;

    npuBuildMpuLoadCmd(
        &cmd, MPU_DEVICE_ID, dst_local_addr, (uint32_t)src_spm_addr, valid_m,
        valid_n, valid_k, layout_mode, sync_indicator, set_completion_sync);
    cmd.launchCmd();
}

static inline void
launch_mpu_load(uint32_t dst_local_addr, uintptr_t src_spm_addr,
                uint32_t valid_m, uint32_t valid_n, uint32_t valid_k,
                uint32_t sync_indicator, uint32_t set_completion_sync)
{
    launch_mpu_load_ext(
        dst_local_addr, src_spm_addr, valid_m, valid_n, valid_k,
        MPU_LAYOUT_MODE_NORMAL, sync_indicator, set_completion_sync);
}

static inline void
launch_mpu_compute(uint32_t src_local_addr_a, uint32_t src_local_addr_b,
                   uint32_t dst_local_addr_c, uint32_t valid_m,
                   uint32_t valid_n, uint32_t valid_k, uint32_t subop,
                   uint32_t sync_indicator, uint32_t set_completion_sync)
{
    NpuCmd cmd;

    npuBuildMpuComputeCmd(
        &cmd, MPU_DEVICE_ID, src_local_addr_a, src_local_addr_b,
        dst_local_addr_c, valid_m, valid_n, valid_k, subop, 0U,
        sync_indicator, set_completion_sync);
    cmd.launchCmd();
}

static inline void
launch_mpu_store_ext(uint32_t src_local_addr, uintptr_t dst_spm_addr,
                     uint32_t valid_m, uint32_t valid_n, uint32_t valid_k,
                     uint32_t layout_mode, uint32_t sync_indicator,
                     uint32_t set_completion_sync)
{
    NpuCmd cmd;

    npuBuildMpuStoreCmd(
        &cmd, MPU_DEVICE_ID, src_local_addr, (uint32_t)dst_spm_addr, valid_m,
        valid_n, valid_k, layout_mode, sync_indicator, set_completion_sync);
    cmd.launchCmd();
}

static inline void
launch_mpu_store(uint32_t src_local_addr, uintptr_t dst_spm_addr,
                 uint32_t valid_m, uint32_t valid_n, uint32_t valid_k,
                 uint32_t sync_indicator, uint32_t set_completion_sync)
{
    launch_mpu_store_ext(
        src_local_addr, dst_spm_addr, valid_m, valid_n, valid_k,
        MPU_LAYOUT_MODE_NORMAL, sync_indicator, set_completion_sync);
}

static inline void
launch_mpu_tensor_loop_ext(uint32_t base_local_addr_a,
                           uint32_t base_local_addr_b,
                           uint32_t base_local_addr_c,
                           uintptr_t base_spm_addr_a,
                           uintptr_t base_spm_addr_b,
                           uintptr_t base_spm_addr_c,
                           uint32_t valid_m, uint32_t valid_n,
                           uint32_t valid_k, uint32_t subop,
                           uint32_t outer_axis, uint32_t outer_count,
                           uint32_t inner_axis, uint32_t inner_count,
                           uint32_t outer_step_tiles,
                           uint32_t inner_step_tiles,
                           uint32_t pingpong_a_enable,
                           uint32_t pingpong_b_enable,
                           uint32_t pingpong_c_enable,
                           uint32_t auto_load_c_for_acc,
                           uint32_t layout_a_skewed,
                           uint32_t layout_b_skewed,
                           uint32_t layout_c_skewed,
                           uint32_t sync_indicator,
                           uint32_t set_completion_sync)
{
    NpuCmd cmd;
    const uint32_t step_cfg = mpuBuildStepCfg(
        outer_step_tiles, inner_step_tiles, pingpong_a_enable,
        pingpong_b_enable, pingpong_c_enable, auto_load_c_for_acc,
        layout_a_skewed, layout_b_skewed, layout_c_skewed);

    npuBuildMpuTensorLoopCmd(
        &cmd, MPU_DEVICE_ID, base_local_addr_a, base_local_addr_b,
        base_local_addr_c, (uint32_t)base_spm_addr_a,
        (uint32_t)base_spm_addr_b, (uint32_t)base_spm_addr_c, valid_m,
        valid_n, valid_k, subop, outer_axis, outer_count, inner_axis,
        inner_count, step_cfg, sync_indicator, set_completion_sync);
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
    write_i8_tile(LOAD_A_SRC, kATile, kATileBytes);
    launch_mpu_load(
        MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK, 1U, 0U);
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_load_c(void)
{
    write_i32_tile(LOAD_C_SRC, kColdTile, kCTileElems);
    launch_mpu_load(
        MPU_LOCAL_ADDR_C0, LOAD_C_SRC, kValidM, kValidN, kValidK, 2U, 0U);
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_store_basic(void)
{
    clear_bytes(STORE_C_DST, kCTileBytes);
    write_i32_tile(LOAD_C_SRC, kColdTile, kCTileElems);

    launch_mpu_load(
        MPU_LOCAL_ADDR_C0, LOAD_C_SRC, kValidM, kValidN, kValidK, 3U, 0U);
    launch_mpu_store(
        MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN, kValidK, 4U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(STORE_C_DST, kColdTile, kCTileElems) ? 0 : 1;
}

static int
scenario_matmul_basic(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(STORE_C_DST, kCTileBytes);
    write_i8_tile(LOAD_A_SRC, kATile, kATileBytes);
    write_i8_tile(LOAD_B_SRC, kBTile, kBTileBytes);
    reference_matmul(
        kATile, kBTile, NULL, expected, kValidM, kValidN, kValidK, 0);

    launch_mpu_load(
        MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK, 5U, 0U);
    launch_mpu_load(
        MPU_LOCAL_ADDR_B0, LOAD_B_SRC, kValidM, kValidN, kValidK, 6U, 0U);
    launch_mpu_compute(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0, kValidM,
        kValidN, kValidK, MPU_SUBOP_MATMUL, 7U, 0U);
    launch_mpu_store(
        MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN, kValidK, 8U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(STORE_C_DST, expected, kCTileElems) ? 0 : 1;
}

static int
scenario_matmul_acc_basic(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(STORE_C_DST, kCTileBytes);
    write_i8_tile(LOAD_A_SRC, kATile, kATileBytes);
    write_i8_tile(LOAD_B_SRC, kBTile, kBTileBytes);
    write_i32_tile(LOAD_C_SRC, kColdTile, kCTileElems);
    reference_matmul(
        kATile, kBTile, kColdTile, expected, kValidM, kValidN, kValidK, 1);

    launch_mpu_load(
        MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK, 9U, 0U);
    launch_mpu_load(
        MPU_LOCAL_ADDR_B0, LOAD_B_SRC, kValidM, kValidN, kValidK, 10U, 0U);
    launch_mpu_load(
        MPU_LOCAL_ADDR_C0, LOAD_C_SRC, kValidM, kValidN, kValidK, 11U, 0U);
    launch_mpu_compute(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0, kValidM,
        kValidN, kValidK, MPU_SUBOP_MATMUL_ACC, 12U, 0U);
    launch_mpu_store(
        MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN, kValidK, 13U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(STORE_C_DST, expected, kCTileElems) ? 0 : 1;
}

static int
scenario_sync_completion(void)
{
    clear_bytes(STORE_C_DST, kCTileBytes);
    write_i32_tile(LOAD_C_SRC, kColdTile, kCTileElems);

    launch_mpu_load(
        MPU_LOCAL_ADDR_C0, LOAD_C_SRC, kValidM, kValidN, kValidK, 14U, 1U);
    npu_launch_sync_wait(MPU_DEVICE_ID, 14U, 0U, 0U, 0U);
    launch_mpu_store(
        MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN, kValidK, 15U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(STORE_C_DST, kColdTile, kCTileElems) ? 0 : 1;
}

static int
scenario_tensor_loop_mn_basic(void)
{
    int32_t expected[kCTileElems];
    const uintptr_t a_tile1_addr = TENSOR_A_BASE + (2U * kATileBytes);
    const uintptr_t b_tile1_addr = TENSOR_B_BASE + kBTileBytes;

    clear_bytes(TENSOR_C_BASE, 4U * kCTileBytes);
    write_i8_tile(TENSOR_A_BASE, kTensorATile0, kATileBytes);
    write_i8_tile(a_tile1_addr, kTensorATile1, kATileBytes);
    write_i8_tile(TENSOR_B_BASE, kTensorBTile0, kBTileBytes);
    write_i8_tile(b_tile1_addr, kTensorBTile1, kBTileBytes);

    launch_mpu_tensor_loop_ext(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0,
        TENSOR_A_BASE, TENSOR_B_BASE, TENSOR_C_BASE, kValidM, kValidN,
        kValidK, MPU_SUBOP_MATMUL, MPU_AXIS_M, 2U, MPU_AXIS_N, 2U, 2U, 1U,
        0U, 0U, 0U, 0U, 0U, 0U, 0U, 16U, 0U);
    npu_cmd_sync_done();

    reference_matmul(
        kTensorATile0, kTensorBTile0, NULL, expected, kValidM, kValidN,
        kValidK, 0);
    if (!verify_i32_tile(TENSOR_C_BASE + (0U * kCTileBytes), expected,
                         kCTileElems)) {
        return 1;
    }
    reference_matmul(
        kTensorATile0, kTensorBTile1, NULL, expected, kValidM, kValidN,
        kValidK, 0);
    if (!verify_i32_tile(TENSOR_C_BASE + (1U * kCTileBytes), expected,
                         kCTileElems)) {
        return 1;
    }
    reference_matmul(
        kTensorATile1, kTensorBTile0, NULL, expected, kValidM, kValidN,
        kValidK, 0);
    if (!verify_i32_tile(TENSOR_C_BASE + (2U * kCTileBytes), expected,
                         kCTileElems)) {
        return 1;
    }
    reference_matmul(
        kTensorATile1, kTensorBTile1, NULL, expected, kValidM, kValidN,
        kValidK, 0);
    if (!verify_i32_tile(TENSOR_C_BASE + (3U * kCTileBytes), expected,
                         kCTileElems)) {
        return 1;
    }

    return 0;
}

static int
scenario_tensor_loop_k_matmul_basic(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(LOOP_K_C_BASE, kCTileBytes);
    write_i8_tile(LOOP_K_A_BASE + 0U * kATileBytes, kLoopKATile0, kATileBytes);
    write_i8_tile(LOOP_K_A_BASE + 1U * kATileBytes, kLoopKATile1, kATileBytes);
    write_i8_tile(LOOP_K_B_BASE + 0U * kBTileBytes, kLoopKBTile0, kBTileBytes);
    write_i8_tile(LOOP_K_B_BASE + 1U * kBTileBytes, kLoopKBTile1, kBTileBytes);
    reference_two_stage_acc(
        kLoopKATile0, kLoopKBTile0, kLoopKATile1, kLoopKBTile1, expected);

    launch_mpu_tensor_loop_ext(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0,
        LOOP_K_A_BASE, LOOP_K_B_BASE, LOOP_K_C_BASE, kValidM, kValidN,
        kValidK, MPU_SUBOP_MATMUL, MPU_AXIS_N, 1U, MPU_AXIS_K, 2U, 1U, 1U,
        0U, 0U, 0U, 0U, 0U, 0U, 0U, 17U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(LOOP_K_C_BASE, expected, kCTileElems) ? 0 : 1;
}

static int
scenario_tensor_loop_k_matmul_acc_header_overwrite_old_c(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(LOOP_K_C_BASE, kCTileBytes);
    write_i8_tile(LOOP_K_A_BASE + 0U * kATileBytes, kLoopKATile0, kATileBytes);
    write_i8_tile(LOOP_K_A_BASE + 1U * kATileBytes, kLoopKATile1, kATileBytes);
    write_i8_tile(LOOP_K_B_BASE + 0U * kBTileBytes, kLoopKBTile0, kBTileBytes);
    write_i8_tile(LOOP_K_B_BASE + 1U * kBTileBytes, kLoopKBTile1, kBTileBytes);
    write_i32_tile(LOOP_K_C_BASE, kColdTile, kCTileElems);
    reference_two_stage_acc(
        kLoopKATile0, kLoopKBTile0, kLoopKATile1, kLoopKBTile1, expected);

    launch_mpu_tensor_loop_ext(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0,
        LOOP_K_A_BASE, LOOP_K_B_BASE, LOOP_K_C_BASE, kValidM, kValidN,
        kValidK, MPU_SUBOP_MATMUL_ACC, MPU_AXIS_N, 1U, MPU_AXIS_K, 2U, 1U,
        1U, 0U, 0U, 0U, 1U, 0U, 0U, 0U, 18U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(LOOP_K_C_BASE, expected, kCTileElems) ? 0 : 1;
}

static int
scenario_tensor_loop_pingpong_ab(void)
{
    int32_t expected[kCTileElems];
    const uintptr_t a_tile1_addr = TENSOR_A_BASE + (2U * kATileBytes);
    const uintptr_t b_tile1_addr = TENSOR_B_BASE + kBTileBytes;

    clear_bytes(TENSOR_C_BASE, 4U * kCTileBytes);
    write_i8_tile(TENSOR_A_BASE, kTensorATile0, kATileBytes);
    write_i8_tile(a_tile1_addr, kTensorATile1, kATileBytes);
    write_i8_tile(TENSOR_B_BASE, kTensorBTile0, kBTileBytes);
    write_i8_tile(b_tile1_addr, kTensorBTile1, kBTileBytes);

    launch_mpu_tensor_loop_ext(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0,
        TENSOR_A_BASE, TENSOR_B_BASE, TENSOR_C_BASE, kValidM, kValidN,
        kValidK, MPU_SUBOP_MATMUL, MPU_AXIS_M, 2U, MPU_AXIS_N, 2U, 2U, 1U,
        1U, 1U, 0U, 0U, 0U, 0U, 0U, 19U, 0U);
    npu_cmd_sync_done();

    reference_matmul(
        kTensorATile1, kTensorBTile1, NULL, expected, kValidM, kValidN,
        kValidK, 0);
    return verify_i32_tile(TENSOR_C_BASE + (3U * kCTileBytes), expected,
                           kCTileElems)
               ? 0
               : 1;
}

static int
scenario_tensor_loop_pingpong_c_per_output_tile(void)
{
    int32_t expected[kCTileElems];
    const uintptr_t b_tile1_addr = TENSOR_B_BASE + kBTileBytes;

    clear_bytes(PP_C_BASE, 2U * kCTileBytes);
    write_i8_tile(TENSOR_A_BASE, kTensorATile0, kATileBytes);
    write_i8_tile(TENSOR_B_BASE, kTensorBTile0, kBTileBytes);
    write_i8_tile(b_tile1_addr, kTensorBTile1, kBTileBytes);

    launch_mpu_tensor_loop_ext(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0,
        TENSOR_A_BASE, TENSOR_B_BASE, PP_C_BASE, kValidM, kValidN, kValidK,
        MPU_SUBOP_MATMUL, MPU_AXIS_M, 1U, MPU_AXIS_N, 2U, 0U, 1U, 0U, 0U,
        1U, 0U, 0U, 0U, 0U, 20U, 0U);
    npu_cmd_sync_done();

    reference_matmul(
        kTensorATile0, kTensorBTile0, NULL, expected, kValidM, kValidN,
        kValidK, 0);
    if (!verify_i32_tile(PP_C_BASE + (0U * kCTileBytes), expected,
                         kCTileElems)) {
        return 1;
    }
    reference_matmul(
        kTensorATile0, kTensorBTile1, NULL, expected, kValidM, kValidN,
        kValidK, 0);
    return verify_i32_tile(PP_C_BASE + (1U * kCTileBytes), expected,
                           kCTileElems)
               ? 0
               : 1;
}

static int
scenario_nonzero_local_offset_load_store(void)
{
    const uint32_t local_c = compose_local_addr(MPU_LOCAL_ADDR_C0, 4U);

    clear_bytes(OFFSET_C_DST, kCTileBytes);
    write_i32_tile(OFFSET_C_SRC, kColdTile, kCTileElems);
    launch_mpu_load_ext(
        local_c, OFFSET_C_SRC, kValidM, kValidN, kValidK,
        MPU_LAYOUT_MODE_NORMAL, 21U, 0U);
    launch_mpu_store_ext(
        local_c, OFFSET_C_DST, kValidM, kValidN, kValidK,
        MPU_LAYOUT_MODE_NORMAL, 22U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(OFFSET_C_DST, kColdTile, kCTileElems) ? 0 : 1;
}

static int
scenario_nonzero_local_offset_compute(void)
{
    const uint32_t local_a = compose_local_addr(MPU_LOCAL_ADDR_A0, 1U);
    const uint32_t local_b = compose_local_addr(MPU_LOCAL_ADDR_B0, 2U);
    const uint32_t local_c = compose_local_addr(MPU_LOCAL_ADDR_C0, 4U);
    int32_t expected[kCTileElems];

    clear_bytes(OFFSET_C_DST, kCTileBytes);
    write_i8_tile(OFFSET_A_SRC, kATile, kATileBytes);
    write_i8_tile(OFFSET_B_SRC, kBTile, kBTileBytes);
    reference_matmul(
        kATile, kBTile, NULL, expected, kValidM, kValidN, kValidK, 0);

    launch_mpu_load_ext(
        local_a, OFFSET_A_SRC, kValidM, kValidN, kValidK,
        MPU_LAYOUT_MODE_NORMAL, 23U, 0U);
    launch_mpu_load_ext(
        local_b, OFFSET_B_SRC, kValidM, kValidN, kValidK,
        MPU_LAYOUT_MODE_NORMAL, 24U, 0U);
    launch_mpu_compute(
        local_a, local_b, local_c, kValidM, kValidN, kValidK,
        MPU_SUBOP_MATMUL, 25U, 0U);
    launch_mpu_store_ext(
        local_c, OFFSET_C_DST, kValidM, kValidN, kValidK,
        MPU_LAYOUT_MODE_NORMAL, 26U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(OFFSET_C_DST, expected, kCTileElems) ? 0 : 1;
}

static int
scenario_skewed_layout_roundtrip(void)
{
    clear_bytes(OFFSET_C_DST, kCTileBytes);
    write_i32_tile(OFFSET_C_SRC, kColdTile, kCTileElems);
    launch_mpu_load_ext(
        MPU_LOCAL_ADDR_C0, OFFSET_C_SRC, kValidM, kValidN, kValidK,
        MPU_LAYOUT_MODE_SKEWED, 27U, 0U);
    launch_mpu_store_ext(
        MPU_LOCAL_ADDR_C0, OFFSET_C_DST, kValidM, kValidN, kValidK,
        MPU_LAYOUT_MODE_SKEWED, 28U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(OFFSET_C_DST, kColdTile, kCTileElems) ? 0 : 1;
}

static int
scenario_local_bank_conflict_stall_stats(void)
{
    return scenario_nonzero_local_offset_compute();
}

static int
scenario_spm_backpressure_stall_stats(void)
{
    return scenario_tensor_loop_k_matmul_acc_header_overwrite_old_c();
}

static int
scenario_tensor_loop_stats_latency(void)
{
    return scenario_tensor_loop_k_matmul_basic();
}

static int
scenario_dma_chain(void)
{
    int32_t expected[kCTileElems];

    clear_bytes(STORE_C_DST, kCTileBytes);
    clear_bytes(DRAM_C_DST, kCTileBytes);
    write_i8_tile(DRAM_A_SRC, kATile, kATileBytes);
    write_i8_tile(DRAM_B_SRC, kBTile, kBTileBytes);
    reference_matmul(
        kATile, kBTile, NULL, expected, kValidM, kValidN, kValidK, 0);

    launch_dma_move_layout(DRAM_A_SRC, LOAD_A_SRC, kATileBytes, 41U, 1U);
    launch_dma_move_layout(DRAM_B_SRC, LOAD_B_SRC, kBTileBytes, 42U, 1U);
    npu_launch_sync_wait(DMA_DEVICE_ID, 41U, 0U, 0U, 0U);
    npu_launch_sync_wait(DMA_DEVICE_ID, 42U, 0U, 0U, 0U);
    launch_mpu_load(
        MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK, 43U, 0U);
    launch_mpu_load(
        MPU_LOCAL_ADDR_B0, LOAD_B_SRC, kValidM, kValidN, kValidK, 44U, 0U);
    launch_mpu_compute(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0, kValidM,
        kValidN, kValidK, MPU_SUBOP_MATMUL, 45U, 0U);
    launch_mpu_store(
        MPU_LOCAL_ADDR_C0, STORE_C_DST, kValidM, kValidN, kValidK, 46U, 1U);
    npu_launch_sync_wait(MPU_DEVICE_ID, 46U, 0U, 0U, 0U);
    launch_dma_move_layout(STORE_C_DST, DRAM_C_DST, (uint32_t)kCTileBytes,
                           47U, 0U);
    npu_cmd_sync_done();

    return verify_i32_tile(DRAM_C_DST, expected, kCTileElems) ? 0 : 1;
}

static int
scenario_invalid_local_addr(void)
{
    NpuCmd cmd;

    write_i8_tile(LOAD_A_SRC, kATile, kATileBytes);
    npuBuildMpuLoadCmd(
        &cmd, MPU_DEVICE_ID, 0x00000180U, (uint32_t)LOAD_A_SRC, kValidM,
        kValidN, kValidK, MPU_LAYOUT_MODE_NORMAL, 60U, 0U);
    cmd.launchCmd();
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_matmul_acc_missing_c(void)
{
    write_i8_tile(LOAD_A_SRC, kATile, kATileBytes);
    write_i8_tile(LOAD_B_SRC, kBTile, kBTileBytes);

    launch_mpu_load(
        MPU_LOCAL_ADDR_A0, LOAD_A_SRC, kValidM, kValidN, kValidK, 61U, 0U);
    launch_mpu_load(
        MPU_LOCAL_ADDR_B0, LOAD_B_SRC, kValidM, kValidN, kValidK, 62U, 0U);
    launch_mpu_compute(
        MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0, MPU_LOCAL_ADDR_C0, kValidM,
        kValidN, kValidK, MPU_SUBOP_MATMUL_ACC, 63U, 0U);
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_tensor_loop_auto_load_c_for_acc_required(void)
{
    NpuCmd cmd;
    const uint32_t step_cfg = mpuBuildStepCfg(
        1U, 1U, 0U, 0U, 0U, 0U, 0U, 0U, 0U);

    npuBuildMpuTensorLoopCmd(
        &cmd, MPU_DEVICE_ID, MPU_LOCAL_ADDR_A0, MPU_LOCAL_ADDR_B0,
        MPU_LOCAL_ADDR_C0, (uint32_t)LOOP_K_A_BASE, (uint32_t)LOOP_K_B_BASE,
        (uint32_t)LOOP_K_C_BASE, kValidM, kValidN, kValidK,
        MPU_SUBOP_MATMUL_ACC, MPU_AXIS_N, 1U, MPU_AXIS_K, 2U, step_cfg,
        64U, 0U);
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
    if (strcmp(argv[1], "store_basic") == 0) {
        return scenario_store_basic();
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
    if (strcmp(argv[1], "tensor_loop_mn_basic") == 0) {
        return scenario_tensor_loop_mn_basic();
    }
    if (strcmp(argv[1], "tensor_loop_k_matmul_basic") == 0) {
        return scenario_tensor_loop_k_matmul_basic();
    }
    if (strcmp(argv[1], "tensor_loop_k_matmul_acc_header_overwrite_old_c") == 0) {
        return scenario_tensor_loop_k_matmul_acc_header_overwrite_old_c();
    }
    if (strcmp(argv[1], "tensor_loop_pingpong_ab") == 0) {
        return scenario_tensor_loop_pingpong_ab();
    }
    if (strcmp(argv[1], "tensor_loop_pingpong_c_per_output_tile") == 0) {
        return scenario_tensor_loop_pingpong_c_per_output_tile();
    }
    if (strcmp(argv[1], "nonzero_local_offset_load_store") == 0) {
        return scenario_nonzero_local_offset_load_store();
    }
    if (strcmp(argv[1], "nonzero_local_offset_compute") == 0) {
        return scenario_nonzero_local_offset_compute();
    }
    if (strcmp(argv[1], "skewed_layout_roundtrip") == 0) {
        return scenario_skewed_layout_roundtrip();
    }
    if (strcmp(argv[1], "local_bank_conflict_stall_stats") == 0) {
        return scenario_local_bank_conflict_stall_stats();
    }
    if (strcmp(argv[1], "spm_backpressure_stall_stats") == 0) {
        return scenario_spm_backpressure_stall_stats();
    }
    if (strcmp(argv[1], "tensor_loop_stats_latency") == 0) {
        return scenario_tensor_loop_stats_latency();
    }
    if (strcmp(argv[1], "dma_chain") == 0) {
        return scenario_dma_chain();
    }
    if (strcmp(argv[1], "invalid_local_addr") == 0) {
        return scenario_invalid_local_addr();
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
