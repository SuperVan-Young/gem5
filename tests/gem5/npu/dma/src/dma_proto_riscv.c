#include <stdint.h>
#include <string.h>

#include "npu_sync.hh"

#define DMA_DEVICE_ID 0x0U

#define DMA_MODE_MOVE_LAYOUT 0x0U
#define DMA_MODE_TRANSPOSE 0x1U
#define DMA_MODE_FILL 0x2U

#define DMA_MEM_SPACE_DRAM 0x0U
#define DMA_MEM_SPACE_SPM 0x1U

#define DMA_CUT_DIM_H 0x0U
#define DMA_CUT_DIM_W 0x1U
#define DMA_CUT_DIM_C 0x2U
#define DMA_DIM_RESERVED 0x3U

#define DRAM_BASE 0x20000000UL
#define SPM_BASE 0x60000000UL

#define SRC_DRAM0 (DRAM_BASE + 0x1000UL)
#define DST_DRAM0 (DRAM_BASE + 0x3000UL)
#define DST_DRAM1 (DRAM_BASE + 0x5000UL)
#define SRC_SPM0 (SPM_BASE + 0x1000UL)
#define DST_SPM0 (SPM_BASE + 0x3000UL)

#define MAX_POLL_ITERS 2000000UL

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
        return linear;
      default:
        return linear;
    }
}

static inline uint8_t
pattern(uint32_t y, uint32_t x, uint32_t z)
{
    return (uint8_t)((y * 37U + x * 11U + z) & 0xffU);
}

static void
fill_tensor(uintptr_t base, Layout layout)
{
    for (uint32_t y = 0; y < layout.h; ++y) {
        for (uint32_t x = 0; x < layout.w; ++x) {
            for (uint32_t z = 0; z < layout.c; ++z) {
                *(volatile uint8_t *)coord_addr(base, layout, y, x, z) =
                    pattern(y, x, z);
            }
        }
    }
}

static void
clear_region(uintptr_t base, size_t bytes)
{
    for (size_t i = 0; i < bytes; ++i) {
        *(volatile uint8_t *)(base + i) = 0;
    }
}

static inline size_t
tensor_bytes(Layout layout)
{
    return (size_t)layout.h * layout.w * layout.c;
}

static int
verify_tensor(uintptr_t base, Layout layout)
{
    for (uint32_t y = 0; y < layout.h; ++y) {
        for (uint32_t x = 0; x < layout.w; ++x) {
            for (uint32_t z = 0; z < layout.c; ++z) {
                volatile uint8_t *ptr =
                    (volatile uint8_t *)coord_addr(base, layout, y, x, z);
                if (*ptr != pattern(y, x, z)) {
                    return 0;
                }
            }
        }
    }
    return 1;
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
fill_bank_cfg(uint32_t dst_bank_id)
{
    return dst_bank_id & 0xfU;
}

static void
build_dma_cmd_raw(NpuCmd *cmd, uintptr_t src_base,
                  uintptr_t dst_base, Layout src_layout,
                  Layout dst_layout, uint32_t data_type,
                  uint32_t mode, uint32_t mode_cfg,
                  uint32_t bank_cfg, uint32_t sync_idx,
                  uint32_t set_completion_sync, uint32_t word15)
{
    const uint32_t op =
        ((data_type & 0x7U) << 5) | ((mode & 0x7U) << 2);

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

static void
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

static void
build_move_layout_cmd(NpuCmd *cmd, uintptr_t src_base, uintptr_t dst_base,
                      Layout src_layout, Layout dst_layout,
                      uint32_t sync_idx, uint32_t set_completion_sync)
{
    build_move_layout_cmd_with_data_type(cmd, src_base, dst_base, src_layout,
                                         dst_layout, 0U, sync_idx,
                                         set_completion_sync);
}

static void
launch_move_layout(uintptr_t src_base, uintptr_t dst_base, Layout src_layout,
                   Layout dst_layout, uint32_t sync_idx,
                   uint32_t set_completion_sync)
{
    NpuCmd cmd;

    build_move_layout_cmd(&cmd, src_base, dst_base, src_layout, dst_layout,
                          sync_idx, set_completion_sync);
    cmd.launchCmd();
}

static void
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

static void
build_fill_cmd(NpuCmd *cmd, Layout dst_layout, uint32_t dst_bank_id,
               uint32_t sync_idx, uint32_t set_completion_sync)
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
    cmd->setWord(2U, 0U);
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
    cmd->setWord(13U, 0U);
    cmd->setWord(14U, fill_bank_cfg(dst_bank_id));
    cmd->setWord(15U, 0U);
}

static void
launch_fill(Layout dst_layout, uint32_t dst_bank_id, uint32_t sync_idx,
            uint32_t set_completion_sync)
{
    NpuCmd cmd;

    build_fill_cmd(&cmd, dst_layout, dst_bank_id, sync_idx,
                   set_completion_sync);
    cmd.launchCmd();
}

static int
poll_until_match(uintptr_t base, Layout layout)
{
    for (unsigned long iter = 0; iter < MAX_POLL_ITERS; ++iter) {
        if (verify_tensor(base, layout)) {
            return 1;
        }
        asm volatile("" ::: "memory");
    }
    return 0;
}

static Layout
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

static Layout
make_layout(uint32_t h, uint32_t w, uint32_t c, uint16_t k)
{
    return make_layout_with_cut(h, w, c, k, DMA_CUT_DIM_W);
}

static int
scenario_basic_dram_to_spm(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, tensor_bytes(layout));
    fill_tensor(SRC_DRAM0, layout);
    launch_move_layout(SRC_DRAM0, DST_SPM0, layout, layout, 5, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_basic_spm_to_dram(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_DRAM0, tensor_bytes(layout));
    fill_tensor(SRC_SPM0, layout);
    launch_move_layout(SRC_SPM0, DST_DRAM0, layout, layout, 6, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_DRAM0, layout) ? 0 : 1;
}

static int
scenario_spm_to_spm(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, tensor_bytes(layout));
    fill_tensor(SRC_SPM0, layout);
    launch_move_layout(SRC_SPM0, DST_SPM0, layout, layout, 10, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_dram_to_dram(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_DRAM1, tensor_bytes(layout));
    fill_tensor(SRC_DRAM0, layout);
    launch_move_layout(SRC_DRAM0, DST_DRAM1, layout, layout, 13, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_DRAM1, layout) ? 0 : 1;
}

static int
scenario_hwc_to_blocked(void)
{
    Layout src = make_layout(2, 4, 8, 0);
    Layout dst = make_layout(2, 4, 8, 2);
    clear_region(DST_SPM0, tensor_bytes(dst));
    fill_tensor(SRC_DRAM0, src);
    launch_move_layout(SRC_DRAM0, DST_SPM0, src, dst, 7, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, dst) ? 0 : 1;
}

static int
scenario_blocked_to_blocked(void)
{
    Layout src = make_layout(2, 8, 4, 2);
    Layout dst = make_layout(2, 8, 4, 4);
    clear_region(DST_DRAM0, tensor_bytes(dst));
    fill_tensor(SRC_SPM0, src);
    launch_move_layout(SRC_SPM0, DST_DRAM0, src, dst, 8, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_DRAM0, dst) ? 0 : 1;
}

static int
scenario_buffer_size_forces_batching(void)
{
    Layout layout = make_layout(2, 8, 8, 0);
    clear_region(DST_SPM0, tensor_bytes(layout));
    fill_tensor(SRC_DRAM0, layout);
    launch_move_layout(SRC_DRAM0, DST_SPM0, layout, layout, 9, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_sync_completion(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, tensor_bytes(layout));
    clear_region(DST_DRAM1, tensor_bytes(layout));
    fill_tensor(SRC_DRAM0, layout);

    launch_move_layout(SRC_DRAM0, DST_SPM0, layout, layout, 11, 1);
    npu_launch_sync_wait(DMA_DEVICE_ID, 11, 0, 0, 0);
    launch_move_layout(DST_SPM0, DST_DRAM1, layout, layout, 12, 0);
    npu_cmd_sync_done();

    return poll_until_match(DST_DRAM1, layout) ? 0 : 1;
}

static int
scenario_queued_chain(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, tensor_bytes(layout));
    clear_region(DST_DRAM1, tensor_bytes(layout));
    fill_tensor(SRC_DRAM0, layout);

    launch_move_layout(SRC_DRAM0, DST_SPM0, layout, layout, 21, 0);
    launch_move_layout(DST_SPM0, DST_DRAM1, layout, layout, 22, 0);
    npu_cmd_sync_done();

    return poll_until_match(DST_DRAM1, layout) ? 0 : 1;
}

static int
scenario_cut_dim_h(void)
{
    Layout layout = make_layout_with_cut(4, 3, 2, 2, DMA_CUT_DIM_H);
    clear_region(DST_SPM0, tensor_bytes(layout));
    fill_tensor(SRC_DRAM0, layout);
    launch_move_layout(SRC_DRAM0, DST_SPM0, layout, layout, 31, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_cut_dim_w(void)
{
    Layout layout = make_layout_with_cut(2, 8, 4, 2, DMA_CUT_DIM_W);
    clear_region(DST_SPM0, tensor_bytes(layout));
    fill_tensor(SRC_DRAM0, layout);
    launch_move_layout(SRC_DRAM0, DST_SPM0, layout, layout, 32, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_cut_dim_c(void)
{
    Layout layout = make_layout_with_cut(2, 4, 8, 2, DMA_CUT_DIM_C);
    clear_region(DST_SPM0, tensor_bytes(layout));
    fill_tensor(SRC_DRAM0, layout);
    launch_move_layout(SRC_DRAM0, DST_SPM0, layout, layout, 33, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_fill_zero_bank(void)
{
    Layout dst = make_layout(2, 4, 8, 0);
    launch_fill(dst, 1U, 40, 1U);
    npu_launch_sync_wait(DMA_DEVICE_ID, 40, 0, 0, 0);
    npu_cmd_sync_done();
    return 0;
}

static int
scenario_fill_invalid_bank_id(void)
{
    Layout dst = make_layout(2, 4, 8, 0);
    launch_fill(dst, 2U, 41, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_fill_reserved_bank_cfg_bits(void)
{
    Layout dst = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_fill_cmd(&cmd, dst, 1U, 42, 0);
    cmd.setWord(14U, fill_bank_cfg(1U) | 0x10U);
    cmd.launchCmd();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_fill_invalid_contract(void)
{
    Layout dst = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_fill_cmd(&cmd, dst, 1U, 43, 0);
    cmd.setWord(1U, (uint32_t)SRC_DRAM0);
    cmd.launchCmd();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_fill_exceeds_bank_size(void)
{
    Layout dst = make_layout(1, 1, 4097, 0);
    launch_fill(dst, 1U, 44, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_invalid_blocked_k_h(void)
{
    Layout src = make_layout_with_cut(3, 4, 8, 2, DMA_CUT_DIM_H);
    Layout dst = make_layout(3, 4, 8, 0);
    launch_move_layout(SRC_DRAM0, DST_SPM0, src, dst, 34, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_invalid_blocked_k_c(void)
{
    Layout src = make_layout_with_cut(2, 4, 6, 4, DMA_CUT_DIM_C);
    Layout dst = make_layout(2, 4, 6, 0);
    launch_move_layout(SRC_DRAM0, DST_SPM0, src, dst, 35, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_memory_space_mismatch(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, SRC_SPM0, DST_DRAM0, layout, layout, 0U,
                      DMA_MODE_MOVE_LAYOUT,
                      move_layout_mode_cfg_raw(DMA_MEM_SPACE_DRAM,
                                               DMA_MEM_SPACE_DRAM,
                                               DMA_CUT_DIM_W,
                                               DMA_CUT_DIM_W),
                      0U, 36, 0, 0U);
    cmd.launchCmd();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_invalid_destination_address(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_move_layout(SRC_DRAM0, 0x10000000UL, layout, layout, 23, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_invalid_blocked_k(void)
{
    Layout src = make_layout(2, 4, 8, 3);
    Layout dst = make_layout(2, 4, 8, 0);
    launch_move_layout(SRC_DRAM0, DST_SPM0, src, dst, 24, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_unsupported_data_type(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_move_layout_with_data_type(SRC_DRAM0, DST_SPM0, layout, layout,
                                     1U, 25, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_reserved_mode(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, SRC_DRAM0, DST_SPM0, layout, layout, 0U, 3U,
                      move_layout_mode_cfg(SRC_DRAM0, DST_SPM0,
                                           DMA_CUT_DIM_W, DMA_CUT_DIM_W),
                      0U, 26, 0, 0U);
    cmd.launchCmd();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_reserved_cut_dim(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, SRC_DRAM0, DST_SPM0, layout, layout, 0U,
                      DMA_MODE_MOVE_LAYOUT,
                      move_layout_mode_cfg(SRC_DRAM0, DST_SPM0,
                                           DMA_DIM_RESERVED, DMA_CUT_DIM_W),
                      0U, 27, 0, 0U);
    cmd.launchCmd();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_reserved_transpose_dim(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, SRC_DRAM0, DST_SPM0, layout, layout, 0U,
                      DMA_MODE_TRANSPOSE,
                      transpose_mode_cfg(SRC_DRAM0, DST_SPM0,
                                         DMA_DIM_RESERVED, DMA_CUT_DIM_W),
                      transpose_bank_cfg(0U, 1U), 28, 0, 0U);
    cmd.launchCmd();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_reserved_bank_cfg_bits(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, SRC_DRAM0, DST_SPM0, layout, layout, 0U,
                      DMA_MODE_TRANSPOSE,
                      transpose_mode_cfg(SRC_DRAM0, DST_SPM0,
                                         DMA_CUT_DIM_H, DMA_CUT_DIM_W),
                      transpose_bank_cfg(0U, 1U) | 0x100U, 29, 0, 0U);
    cmd.launchCmd();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_out_of_range_bank_id(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    NpuCmd cmd;

    build_dma_cmd_raw(&cmd, SRC_DRAM0, DST_SPM0, layout, layout, 0U,
                      DMA_MODE_TRANSPOSE,
                      transpose_mode_cfg(SRC_DRAM0, DST_SPM0,
                                         DMA_CUT_DIM_H, DMA_CUT_DIM_W),
                      transpose_bank_cfg(2U, 1U), 30, 0, 0U);
    cmd.launchCmd();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}


static int
scenario_invalid_address(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_move_layout(0x10000000UL, DST_SPM0, layout, layout, 13, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

int
main(int argc, char **argv)
{
    if (argc != 2) {
        return 2;
    }

    if (strcmp(argv[1], "basic_dram_to_spm") == 0) {
        return scenario_basic_dram_to_spm();
    }
    if (strcmp(argv[1], "basic_spm_to_dram") == 0) {
        return scenario_basic_spm_to_dram();
    }
    if (strcmp(argv[1], "spm_to_spm") == 0) {
        return scenario_spm_to_spm();
    }
    if (strcmp(argv[1], "dram_to_dram") == 0) {
        return scenario_dram_to_dram();
    }
    if (strcmp(argv[1], "hwc_to_blocked") == 0) {
        return scenario_hwc_to_blocked();
    }
    if (strcmp(argv[1], "blocked_to_blocked") == 0) {
        return scenario_blocked_to_blocked();
    }
    if (strcmp(argv[1], "buffer_size_forces_batching") == 0) {
        return scenario_buffer_size_forces_batching();
    }
    if (strcmp(argv[1], "sync_completion") == 0) {
        return scenario_sync_completion();
    }
    if (strcmp(argv[1], "queued_chain") == 0) {
        return scenario_queued_chain();
    }
    if (strcmp(argv[1], "cut_dim_h") == 0) {
        return scenario_cut_dim_h();
    }
    if (strcmp(argv[1], "cut_dim_w") == 0) {
        return scenario_cut_dim_w();
    }
    if (strcmp(argv[1], "cut_dim_c") == 0) {
        return scenario_cut_dim_c();
    }
    if (strcmp(argv[1], "fill_zero_bank") == 0) {
        return scenario_fill_zero_bank();
    }
    if (strcmp(argv[1], "fill_invalid_bank_id") == 0) {
        return scenario_fill_invalid_bank_id();
    }
    if (strcmp(argv[1], "fill_reserved_bank_cfg_bits") == 0) {
        return scenario_fill_reserved_bank_cfg_bits();
    }
    if (strcmp(argv[1], "fill_invalid_contract") == 0) {
        return scenario_fill_invalid_contract();
    }
    if (strcmp(argv[1], "fill_exceeds_bank_size") == 0) {
        return scenario_fill_exceeds_bank_size();
    }
    if (strcmp(argv[1], "invalid_destination_address") == 0) {
        return scenario_invalid_destination_address();
    }
    if (strcmp(argv[1], "invalid_blocked_k") == 0) {
        return scenario_invalid_blocked_k();
    }
    if (strcmp(argv[1], "invalid_blocked_k_h") == 0) {
        return scenario_invalid_blocked_k_h();
    }
    if (strcmp(argv[1], "invalid_blocked_k_c") == 0) {
        return scenario_invalid_blocked_k_c();
    }
    if (strcmp(argv[1], "unsupported_data_type") == 0) {
        return scenario_unsupported_data_type();
    }
    if (strcmp(argv[1], "reserved_mode") == 0) {
        return scenario_reserved_mode();
    }
    if (strcmp(argv[1], "reserved_cut_dim") == 0) {
        return scenario_reserved_cut_dim();
    }
    if (strcmp(argv[1], "reserved_transpose_dim") == 0) {
        return scenario_reserved_transpose_dim();
    }
    if (strcmp(argv[1], "reserved_bank_cfg_bits") == 0) {
        return scenario_reserved_bank_cfg_bits();
    }
    if (strcmp(argv[1], "out_of_range_bank_id") == 0) {
        return scenario_out_of_range_bank_id();
    }
    if (strcmp(argv[1], "memory_space_mismatch") == 0) {
        return scenario_memory_space_mismatch();
    }
    if (strcmp(argv[1], "invalid_address") == 0) {
        return scenario_invalid_address();
    }

    return 3;
}
