#ifndef TESTS_GEM5_NPU_UTILS_DMA_PANIC_CASES_H_
#define TESTS_GEM5_NPU_UTILS_DMA_PANIC_CASES_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cmd/dma.hh"
#include "npu_sync.hh"

/*
 * Shared illegal DMA command builders for reject testcases.
 *
 * Every testcase defines DMA_PANIC_CASE before including this header. The
 * selected macro names exactly one scenario_* function below, and main() only
 * dispatches that single scenario. This keeps the checked-in workload tree
 * "one testcase, one rejection rule" while still avoiding duplicated command
 * construction code.
 */

#define DMA_PANIC_STRINGIFY_INNER(x) #x
#define DMA_PANIC_STRINGIFY(x) DMA_PANIC_STRINGIFY_INNER(x)
#define DMA_PANIC_JOIN_INNER(a, b) a##b
#define DMA_PANIC_JOIN(a, b) DMA_PANIC_JOIN_INNER(a, b)

#ifndef DMA_PANIC_CASE
#error "DMA_PANIC_CASE must be defined before including dma_panic_cases.hh"
#endif

#define DMA_DEVICE_ID 0x0U

#define DMA_MODE_MOVE_LAYOUT 0x0U
#define DMA_MODE_TRANSPOSE 0x1U
#define DMA_MODE_FILL 0x2U

#define DMA_STAGE_LEGACY 0x0U
#define DMA_STAGE_LOAD 0x1U
#define DMA_STAGE_COMPUTE 0x2U
#define DMA_STAGE_STORE 0x3U

#define DMA_MEM_SPACE_DRAM 0x0U
#define DMA_MEM_SPACE_SPM 0x1U
#define DMA_MEM_SPACE_DMA_BANK 0x2U

#define DMA_CUT_DIM_H 0x0U
#define DMA_CUT_DIM_W 0x1U
#define DMA_CUT_DIM_C 0x2U
#define DMA_DIM_RESERVED 0x3U

#define DRAM_BASE 0x20000000UL
#define SPM_BASE 0x60000000UL

typedef struct
{
    uint32_t h;
    uint32_t w;
    uint32_t c;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t stride_c;
    uint16_t k;
    uint32_t cut_dim;
} Layout;

static inline uintptr_t
coord_addr(uintptr_t base, Layout layout, uint32_t y, uint32_t x, uint32_t z)
{
    const uintptr_t linear = base + (uintptr_t)y * layout.stride_h +
                             (uintptr_t)x * layout.stride_w +
                             (uintptr_t)z * layout.stride_c;

    if (layout.k == 0) {
        return linear;
    }

    switch (layout.cut_dim) {
      case DMA_CUT_DIM_H:
        return base + (uintptr_t)(y / layout.k) * layout.stride_w * layout.w +
               (uintptr_t)x * layout.stride_w +
               (uintptr_t)z * layout.stride_c +
               (uintptr_t)(y % layout.k) * layout.stride_h;
      case DMA_CUT_DIM_W:
        return base + (uintptr_t)y * layout.stride_h +
               (uintptr_t)(x / layout.k) * layout.stride_c * layout.c +
               (uintptr_t)z * layout.stride_c +
               (uintptr_t)(x % layout.k) * layout.stride_w;
      case DMA_CUT_DIM_C:
      default:
        return linear;
    }
}

static inline Layout
make_layout_with_cut(uint32_t h, uint32_t w, uint32_t c, uint16_t k,
                     uint32_t cut_dim)
{
    Layout layout;
    layout.h = h;
    layout.w = w;
    layout.c = c;
    layout.k = k;
    layout.cut_dim = cut_dim;

    if (k == 0 || cut_dim == DMA_CUT_DIM_C) {
        layout.stride_c = 1;
        layout.stride_w = c;
        layout.stride_h = w * c;
        return layout;
    }

    if (cut_dim == DMA_CUT_DIM_H) {
        layout.stride_h = 1;
        layout.stride_c = k;
        layout.stride_w = c * k;
        return layout;
    }

    layout.stride_w = 1;
    layout.stride_c = k;
    layout.stride_h = w * c;
    return layout;
}

static inline Layout
make_layout(uint32_t h, uint32_t w, uint32_t c, uint16_t k)
{
    return make_layout_with_cut(h, w, c, k, DMA_CUT_DIM_W);
}

static inline uint32_t
mem_space_for_base(uintptr_t base)
{
    return base >= SPM_BASE ? DMA_MEM_SPACE_SPM : DMA_MEM_SPACE_DRAM;
}

static inline uint32_t
move_layout_mode_cfg_raw(uint32_t src_mem_space, uint32_t dst_mem_space,
                         uint32_t src_cut_dim, uint32_t dst_cut_dim)
{
    return (src_mem_space & 0x1U) | ((dst_mem_space & 0x1U) << 1) |
           ((src_cut_dim & 0x3U) << 2) | ((dst_cut_dim & 0x3U) << 4);
}

static inline uint32_t
move_layout_mode_cfg(uintptr_t src_base, uintptr_t dst_base,
                     uint32_t src_cut_dim, uint32_t dst_cut_dim)
{
    return move_layout_mode_cfg_raw(mem_space_for_base(src_base),
                                    mem_space_for_base(dst_base),
                                    src_cut_dim, dst_cut_dim);
}

static inline uint32_t
transpose_mode_cfg(uintptr_t src_base, uintptr_t dst_base,
                   uint32_t dim_a, uint32_t dim_b)
{
    return (mem_space_for_base(src_base) & 0x1U) |
           ((mem_space_for_base(dst_base) & 0x1U) << 1) |
           ((dim_a & 0x3U) << 2) |
           ((dim_b & 0x3U) << 4);
}

static inline uint32_t
transpose_bank_cfg(uint32_t src_bank_id, uint32_t dst_bank_id)
{
    return (src_bank_id & 0xfU) | ((dst_bank_id & 0xfU) << 4);
}

static inline uint32_t
fill_mode_cfg(uint32_t dst_mem_space)
{
    return dst_mem_space & 0x3U;
}

static inline uint32_t
fill_bank_cfg(uint32_t dst_bank_id)
{
    return dst_bank_id & 0xfU;
}

static inline void
build_dma_cmd_raw_with_stage(NpuCmd *cmd, uintptr_t src_base,
                             uintptr_t dst_base, Layout src_layout,
                             Layout dst_layout, uint32_t data_type,
                             uint32_t mode, uint32_t stage,
                             uint32_t mode_cfg, uint32_t bank_cfg,
                             uint32_t sync_idx,
                             uint32_t set_completion_sync, uint32_t word15)
{
    const uint32_t op =
        ((data_type & 0x7U) << 5) | ((mode & 0x7U) << 2) |
        (stage & 0x3U);

    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_DMA);
    cmd->setDeviceId(DMA_DEVICE_ID);
    cmd->setOpCode(op);
    cmd->setSyncIndicator(sync_idx);
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
    cmd->setWord(14U, bank_cfg);
    cmd->setWord(15U, word15);
}

static inline void
build_dma_cmd_raw(NpuCmd *cmd, uintptr_t src_base, uintptr_t dst_base,
                  Layout src_layout, Layout dst_layout, uint32_t data_type,
                  uint32_t mode, uint32_t mode_cfg, uint32_t bank_cfg,
                  uint32_t sync_idx, uint32_t set_completion_sync,
                  uint32_t word15)
{
    build_dma_cmd_raw_with_stage(
        cmd, src_base, dst_base, src_layout, dst_layout, data_type, mode,
        DMA_STAGE_LEGACY, mode_cfg, bank_cfg, sync_idx,
        set_completion_sync, word15);
}

static inline void
build_move_layout_cmd_with_data_type(NpuCmd *cmd, uintptr_t src_base,
                                     uintptr_t dst_base, Layout src_layout,
                                     Layout dst_layout, uint32_t data_type,
                                     uint32_t sync_idx,
                                     uint32_t set_completion_sync)
{
    build_dma_cmd_raw(
        cmd, src_base, dst_base, src_layout, dst_layout, data_type,
        DMA_MODE_MOVE_LAYOUT,
        move_layout_mode_cfg(src_base, dst_base, src_layout.cut_dim,
                             dst_layout.cut_dim),
        0U, sync_idx, set_completion_sync, 0U);
}

static inline void
build_move_layout_cmd(NpuCmd *cmd, uintptr_t src_base, uintptr_t dst_base,
                      Layout src_layout, Layout dst_layout,
                      uint32_t sync_idx, uint32_t set_completion_sync)
{
    build_move_layout_cmd_with_data_type(cmd, src_base, dst_base, src_layout,
                                         dst_layout, 0U, sync_idx,
                                         set_completion_sync);
}

static inline void
launch_move_layout(uintptr_t src_base, uintptr_t dst_base, Layout src_layout,
                   Layout dst_layout, uint32_t sync_idx,
                   uint32_t set_completion_sync)
{
    NpuCmd cmd;

    build_move_layout_cmd(&cmd, src_base, dst_base, src_layout, dst_layout,
                          sync_idx, set_completion_sync);
    cmd.launchCmd();
}

static inline void
launch_move_layout_with_data_type(uintptr_t src_base, uintptr_t dst_base,
                                  Layout src_layout, Layout dst_layout,
                                  uint32_t data_type, uint32_t sync_idx,
                                  uint32_t set_completion_sync)
{
    NpuCmd cmd;

    build_move_layout_cmd_with_data_type(
        &cmd, src_base, dst_base, src_layout, dst_layout, data_type,
        sync_idx, set_completion_sync);
    cmd.launchCmd();
}

static inline void
build_transpose_cmd(NpuCmd *cmd, uintptr_t src_base, uintptr_t dst_base,
                    Layout src_layout, Layout dst_layout, uint32_t dim_a,
                    uint32_t dim_b, uint32_t src_bank_id,
                    uint32_t dst_bank_id, uint32_t sync_idx,
                    uint32_t set_completion_sync)
{
    build_dma_cmd_raw(cmd, src_base, dst_base, src_layout, dst_layout, 0U,
                      DMA_MODE_TRANSPOSE,
                      transpose_mode_cfg(src_base, dst_base, dim_a, dim_b),
                      transpose_bank_cfg(src_bank_id, dst_bank_id), sync_idx,
                      set_completion_sync, 0U);
}

static inline void
launch_transpose(uintptr_t src_base, uintptr_t dst_base, Layout src_layout,
                 Layout dst_layout, uint32_t dim_a, uint32_t dim_b,
                 uint32_t src_bank_id, uint32_t dst_bank_id,
                 uint32_t sync_idx, uint32_t set_completion_sync)
{
    NpuCmd cmd;

    build_transpose_cmd(&cmd, src_base, dst_base, src_layout, dst_layout,
                        dim_a, dim_b, src_bank_id, dst_bank_id, sync_idx,
                        set_completion_sync);
    cmd.launchCmd();
}

static inline void
build_fill_cmd(NpuCmd *cmd, uintptr_t dst_base, Layout dst_layout,
               uint32_t dst_mem_space, uint32_t dst_bank_id,
               uint8_t fill_value, uint32_t sync_idx,
               uint32_t set_completion_sync)
{
    const uint32_t op = (0U << 5) | (DMA_MODE_FILL << 2);

    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_DMA);
    cmd->setDeviceId(DMA_DEVICE_ID);
    cmd->setOpCode(op);
    cmd->setSyncIndicator(sync_idx);
    cmd->setSetIndicatorSns(set_completion_sync ? 1U : 0U);
    cmd->setSetIndicatorSnd(0U);
    cmd->clearCommonReservedBits();
    cmd->setWord(1U, 0U);
    cmd->setWord(2U, (uint32_t)dst_base);
    cmd->setWord(3U, dst_layout.h);
    cmd->setWord(4U, dst_layout.w);
    cmd->setWord(5U, dst_layout.c);
    cmd->setWord(6U, 0U);
    cmd->setWord(7U, 0U);
    cmd->setWord(8U, 0U);
    cmd->setWord(9U, dst_layout.stride_h);
    cmd->setWord(10U, dst_layout.stride_w);
    cmd->setWord(11U, dst_layout.stride_c);
    cmd->setWord(12U, 0U);
    cmd->setWord(13U, fill_mode_cfg(dst_mem_space));
    cmd->setWord(14U, fill_bank_cfg(dst_bank_id));
    cmd->setWord(15U, fill_value);
}

static inline void
launch_fill(uintptr_t dst_base, Layout dst_layout, uint32_t dst_mem_space,
            uint32_t dst_bank_id, uint8_t fill_value, uint32_t sync_idx,
            uint32_t set_completion_sync)
{
    NpuCmd cmd;

    build_fill_cmd(&cmd, dst_base, dst_layout, dst_mem_space, dst_bank_id,
                   fill_value, sync_idx, set_completion_sync);
    cmd.launchCmd();
}

static inline void
spin_forever(void)
{
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_invalid_source_address(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_move_layout(0x10000000UL, SPM_BASE + 0x3000UL, layout, layout, 13,
                       0);
    spin_forever();
    return 0;
}

static int
scenario_invalid_destination_address(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_move_layout(DRAM_BASE + 0x1000UL, 0x10000000UL, layout, layout, 13,
                       0);
    spin_forever();
    return 0;
}

static int
scenario_invalid_blocked_k(void)
{
    Layout src = make_layout(2, 4, 8, 3);
    Layout dst = make_layout(2, 4, 8, 0);
    launch_move_layout(DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, src, dst,
                       24, 0);
    spin_forever();
    return 0;
}

static int
scenario_invalid_blocked_k_h(void)
{
    Layout src = make_layout_with_cut(3, 4, 8, 2, DMA_CUT_DIM_H);
    Layout dst = make_layout(3, 4, 8, 0);
    launch_move_layout(DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, src, dst,
                       34, 0);
    spin_forever();
    return 0;
}

static int
scenario_invalid_blocked_k_c(void)
{
    Layout src = make_layout_with_cut(2, 4, 6, 4, DMA_CUT_DIM_C);
    Layout dst = make_layout(2, 4, 6, 0);
    launch_move_layout(DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, src, dst,
                       35, 0);
    spin_forever();
    return 0;
}

static int
scenario_unsupported_data_type(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_move_layout_with_data_type(DRAM_BASE + 0x1000UL,
                                      SPM_BASE + 0x3000UL, layout, layout,
                                      1U, 25, 0);
    spin_forever();
    return 0;
}

static int
scenario_reserved_mode(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, layout,
                      layout, 0U, 3U,
                      move_layout_mode_cfg(DRAM_BASE + 0x1000UL,
                                           SPM_BASE + 0x3000UL,
                                           DMA_CUT_DIM_W, DMA_CUT_DIM_W),
                      0U, 26, 0, 0U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_reserved_cut_dim(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, layout,
                      layout, 0U, DMA_MODE_MOVE_LAYOUT,
                      move_layout_mode_cfg(DRAM_BASE + 0x1000UL,
                                           SPM_BASE + 0x3000UL,
                                           DMA_DIM_RESERVED, DMA_CUT_DIM_W),
                      0U, 27, 0, 0U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_reserved_transpose_dim(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, layout,
                      layout, 0U, DMA_MODE_TRANSPOSE,
                      transpose_mode_cfg(DRAM_BASE + 0x1000UL,
                                         SPM_BASE + 0x3000UL,
                                         DMA_DIM_RESERVED, DMA_CUT_DIM_W),
                      transpose_bank_cfg(0U, 1U), 28, 0, 0U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_reserved_bank_cfg_bits(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, layout,
                      layout, 0U, DMA_MODE_TRANSPOSE,
                      transpose_mode_cfg(DRAM_BASE + 0x1000UL,
                                         SPM_BASE + 0x3000UL,
                                         DMA_CUT_DIM_H, DMA_CUT_DIM_W),
                      transpose_bank_cfg(0U, 1U) | 0x100U, 29, 0, 0U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_out_of_range_bank_id(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, layout,
                      layout, 0U, DMA_MODE_TRANSPOSE,
                      transpose_mode_cfg(DRAM_BASE + 0x1000UL,
                                         SPM_BASE + 0x3000UL,
                                         DMA_CUT_DIM_H, DMA_CUT_DIM_W),
                      transpose_bank_cfg(2U, 1U), 30, 0, 0U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_memory_space_mismatch(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, SPM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, layout,
                      layout, 0U, DMA_MODE_MOVE_LAYOUT,
                      move_layout_mode_cfg_raw(DMA_MEM_SPACE_DRAM,
                                               DMA_MEM_SPACE_DRAM,
                                               DMA_CUT_DIM_W,
                                               DMA_CUT_DIM_W),
                      0U, 36, 0, 0U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_fill_invalid_bank_id(void)
{
    Layout dst = make_layout(2, 4, 8, 0);
    launch_fill(0U, dst, DMA_MEM_SPACE_DMA_BANK, 2U, 0U, 41, 0);
    spin_forever();
    return 0;
}

static int
scenario_staged_fill_unsupported(void)
{
    Layout dst = make_layout(1, 1, 64, 0);
    NpuCmd cmd;

    build_dma_cmd_raw_with_stage(
        &cmd, 0U, SPM_BASE + 0x3000UL, dst, dst, 0U, DMA_MODE_FILL,
        DMA_STAGE_LOAD, fill_mode_cfg(DMA_MEM_SPACE_SPM), 0U, 67, 0, 0U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_fill_reserved_bank_cfg_bits(void)
{
    Layout dst = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_fill_cmd(&cmd, 0U, dst, DMA_MEM_SPACE_DMA_BANK, 1U, 0U, 42, 0);
    cmd.setWord(14U, fill_bank_cfg(1U) | 0x10U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_fill_invalid_contract(void)
{
    Layout dst = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_fill_cmd(&cmd, 0U, dst, DMA_MEM_SPACE_DMA_BANK, 1U, 0U, 43, 0);
    cmd.setWord(1U, (uint32_t)(DRAM_BASE + 0x1000UL));
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_fill_exceeds_bank_size(void)
{
    Layout dst = make_layout(1, 1, 4097, 0);
    launch_fill(0U, dst, DMA_MEM_SPACE_DMA_BANK, 1U, 0U, 44, 0);
    spin_forever();
    return 0;
}

static int
scenario_fill_invalid_dst_mem_space(void)
{
    Layout dst = make_layout(1, 1, 64, 0);
    NpuCmd cmd;

    build_fill_cmd(&cmd, DRAM_BASE + 0x3000UL, dst, DMA_MEM_SPACE_DRAM, 0U,
                   0U, 57, 0);
    cmd.setWord(13U, 0x3U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_fill_partial_line_dram(void)
{
    Layout dst = make_layout(1, 1, 32, 0);
    launch_fill(DRAM_BASE + 0x3000UL, dst, DMA_MEM_SPACE_DRAM, 0U, 0U, 58,
                0);
    spin_forever();
    return 0;
}

static int
scenario_fill_partial_line_spm(void)
{
    Layout dst = make_layout(1, 1, 32, 0);
    launch_fill(SPM_BASE + 0x3000UL, dst, DMA_MEM_SPACE_SPM, 0U, 0U, 59, 0);
    spin_forever();
    return 0;
}

static int
scenario_fill_reserved_fill_value_bits(void)
{
    Layout dst = make_layout(1, 1, 64, 0);
    NpuCmd cmd;

    build_fill_cmd(&cmd, 0U, dst, DMA_MEM_SPACE_DMA_BANK, 1U, 0U, 60, 0);
    cmd.setWord(15U, 0x100U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_transpose_same_bank(void)
{
    Layout src = make_layout(2, 4, 1, 0);
    Layout dst = make_layout(4, 2, 1, 0);
    launch_transpose(DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, src, dst,
                     DMA_CUT_DIM_H, DMA_CUT_DIM_W, 1U, 1U, 48, 0);
    spin_forever();
    return 0;
}

static int
scenario_staged_transpose_unsupported(void)
{
    Layout src = make_layout(2, 4, 1, 0);
    Layout dst = make_layout(4, 2, 1, 0);
    NpuCmd cmd;

    build_dma_cmd_raw_with_stage(
        &cmd, DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, src, dst, 0U,
        DMA_MODE_TRANSPOSE, DMA_STAGE_LOAD,
        transpose_mode_cfg(DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL,
                           DMA_CUT_DIM_H, DMA_CUT_DIM_W),
        transpose_bank_cfg(0U, 1U), 68, 0, 0U);
    cmd.launchCmd();
    spin_forever();
    return 0;
}

static int
scenario_transpose_equal_dims(void)
{
    Layout src = make_layout(2, 4, 1, 0);
    Layout dst = make_layout(2, 4, 1, 0);
    launch_transpose(DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, src, dst,
                     DMA_CUT_DIM_H, DMA_CUT_DIM_H, 0U, 1U, 49, 0);
    spin_forever();
    return 0;
}

static int
scenario_transpose_nonzero_k(void)
{
    Layout src = make_layout(2, 4, 1, 2);
    Layout dst = make_layout(4, 2, 1, 0);
    launch_transpose(DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, src, dst,
                     DMA_CUT_DIM_H, DMA_CUT_DIM_W, 0U, 1U, 50, 0);
    spin_forever();
    return 0;
}

static int
scenario_transpose_exceeds_bank_size(void)
{
    Layout src = make_layout(1, 1, 4097, 0);
    Layout dst = make_layout(1, 4097, 1, 0);
    launch_transpose(DRAM_BASE + 0x1000UL, SPM_BASE + 0x3000UL, src, dst,
                     DMA_CUT_DIM_W, DMA_CUT_DIM_C, 0U, 1U, 51, 0);
    spin_forever();
    return 0;
}

static int
dma_panic_dispatch(const char *scenario)
{
    /*
     * The workload binary is built for one reject testcase at a time. The
     * runtime string check keeps the process argv contract explicit and makes
     * accidental cross-case reuse fail closed.
     */
    if (strcmp(scenario, DMA_PANIC_STRINGIFY(DMA_PANIC_CASE)) == 0) {
        return DMA_PANIC_JOIN(scenario_, DMA_PANIC_CASE)();
    }
    return 3;
}

int
main(int argc, char **argv)
{
    if (argc != 2) {
        return 2;
    }

    return dma_panic_dispatch(argv[1]);
}

#endif
