#ifndef TESTS_GEM5_NPU_UTILS_CMD_DMA_H_
#define TESTS_GEM5_NPU_UTILS_CMD_DMA_H_

#include "common.hh"

enum DmaOpcodeMode {
    DMA_MODE_MOVE_LAYOUT = 0x0U,
    DMA_MODE_TRANSPOSE = 0x1U,
    DMA_MODE_FILL = 0x2U,
};

enum DmaCommandStage {
    DMA_CMD_STAGE_LEGACY = 0x0U,
    DMA_CMD_STAGE_LOAD = 0x1U,
    DMA_CMD_STAGE_COMPUTE = 0x2U,
    DMA_CMD_STAGE_STORE = 0x3U,
};

enum DmaMemSpace {
    DMA_MEM_SPACE_DRAM = 0x0U,
    DMA_MEM_SPACE_SPM = 0x1U,
    DMA_MEM_SPACE_DMA_BANK = 0x2U,
};

enum DmaCutDim {
    DMA_CUT_DIM_H = 0x0U,
    DMA_CUT_DIM_W = 0x1U,
    DMA_CUT_DIM_C = 0x2U,
};

enum DmaCmdWord {
    DMA_CMD_WORD_SRC_BASE = 1U,
    DMA_CMD_WORD_DST_BASE = 2U,
    DMA_CMD_WORD_SHAPE_H = 3U,
    DMA_CMD_WORD_SHAPE_W = 4U,
    DMA_CMD_WORD_SHAPE_C = 5U,
    DMA_CMD_WORD_SRC_STRIDE_H = 6U,
    DMA_CMD_WORD_SRC_STRIDE_W = 7U,
    DMA_CMD_WORD_SRC_STRIDE_C = 8U,
    DMA_CMD_WORD_DST_STRIDE_H = 9U,
    DMA_CMD_WORD_DST_STRIDE_W = 10U,
    DMA_CMD_WORD_DST_STRIDE_C = 11U,
    DMA_CMD_WORD_BLOCK_CFG = 12U,
    DMA_CMD_WORD_MODE_CFG = 13U,
    DMA_CMD_WORD_BANK_CFG = 14U,
    DMA_CMD_WORD_WORD15 = 15U,
};

struct DmaCmdInstr
{
    uint32_t header;
    uint32_t src_base_addr;
    uint32_t dst_base_addr;
    uint32_t shape_h;
    uint32_t shape_w;
    uint32_t shape_c;
    uint32_t src_stride_h;
    uint32_t src_stride_w;
    uint32_t src_stride_c;
    uint32_t dst_stride_h;
    uint32_t dst_stride_w;
    uint32_t dst_stride_c;
    uint32_t block_cfg;
    uint32_t mode_cfg;
    uint32_t bank_cfg;
    uint32_t word15;
};

static_assert(sizeof(DmaCmdInstr) == NPU_CMD_BUFFER_BYTES,
              "DMA command struct must remain 64 bytes.");

struct DmaLayout
{
    uint32_t h;
    uint32_t w;
    uint32_t c;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t stride_c;
    uint16_t k;
    uint32_t cut_dim;
};

static inline uint32_t
dma_mem_space_for_base(uintptr_t base)
{
    return base >= 0x60000000ULL ? DMA_MEM_SPACE_SPM : DMA_MEM_SPACE_DRAM;
}

static inline uint32_t
dma_move_layout_mode_cfg(uintptr_t src_base, uintptr_t dst_base,
                         uint32_t src_cut_dim, uint32_t dst_cut_dim)
{
    return (dma_mem_space_for_base(src_base) & 0x1U) |
           ((dma_mem_space_for_base(dst_base) & 0x1U) << 1) |
           ((src_cut_dim & 0x3U) << 2) |
           ((dst_cut_dim & 0x3U) << 4);
}

static inline void
dma_cmd_init_move_layout(NpuCmd *cmd, uint32_t device_id,
                         uintptr_t src_base, uintptr_t dst_base,
                         const DmaLayout *src_layout,
                         const DmaLayout *dst_layout,
                         uint32_t sync_indicator,
                         uint32_t set_completion_sync)
{
    DmaCmdInstr dma_cmd = {};
    NpuCmdBinaryData binary = {};

    dma_cmd.header = npuBuildHeaderWord(
        NPU_DEVICE_TYPE_DMA, device_id,
        (0U << 5) | (DMA_MODE_MOVE_LAYOUT << 2), sync_indicator,
        set_completion_sync ? 1U : 0U, 0U);
    dma_cmd.src_base_addr = (uint32_t)src_base;
    dma_cmd.dst_base_addr = (uint32_t)dst_base;
    dma_cmd.shape_h = src_layout->h;
    dma_cmd.shape_w = src_layout->w;
    dma_cmd.shape_c = src_layout->c;
    dma_cmd.src_stride_h = src_layout->stride_h;
    dma_cmd.src_stride_w = src_layout->stride_w;
    dma_cmd.src_stride_c = src_layout->stride_c;
    dma_cmd.dst_stride_h = dst_layout->stride_h;
    dma_cmd.dst_stride_w = dst_layout->stride_w;
    dma_cmd.dst_stride_c = dst_layout->stride_c;
    dma_cmd.block_cfg =
        ((uint32_t)dst_layout->k << 16) | src_layout->k;
    dma_cmd.mode_cfg = dma_move_layout_mode_cfg(
        src_base, dst_base, src_layout->cut_dim, dst_layout->cut_dim);
    dma_cmd.bank_cfg = 0U;
    dma_cmd.word15 = 0U;
    npuBinaryDataFromObject(&binary, dma_cmd);
    cmd->loadBinary(binary);
}

static inline void
dma_cmd_init_move_layout_stage(NpuCmd *cmd, uint32_t device_id,
                               uintptr_t src_base, uintptr_t dst_base,
                               const DmaLayout *src_layout,
                               const DmaLayout *dst_layout,
                               uint32_t command_stage,
                               uint32_t sync_indicator,
                               uint32_t set_completion_sync)
{
    DmaCmdInstr dma_cmd = {};
    NpuCmdBinaryData binary = {};

    dma_cmd.header = npuBuildHeaderWord(
        NPU_DEVICE_TYPE_DMA, device_id,
        (0U << 5) | (DMA_MODE_MOVE_LAYOUT << 2) | (command_stage & 0x3U),
        sync_indicator, set_completion_sync ? 1U : 0U, 0U);
    dma_cmd.src_base_addr = (uint32_t)src_base;
    dma_cmd.dst_base_addr = (uint32_t)dst_base;
    dma_cmd.shape_h = src_layout->h;
    dma_cmd.shape_w = src_layout->w;
    dma_cmd.shape_c = src_layout->c;
    dma_cmd.src_stride_h = src_layout->stride_h;
    dma_cmd.src_stride_w = src_layout->stride_w;
    dma_cmd.src_stride_c = src_layout->stride_c;
    dma_cmd.dst_stride_h = dst_layout->stride_h;
    dma_cmd.dst_stride_w = dst_layout->stride_w;
    dma_cmd.dst_stride_c = dst_layout->stride_c;
    dma_cmd.block_cfg =
        ((uint32_t)dst_layout->k << 16) | src_layout->k;
    dma_cmd.mode_cfg = dma_move_layout_mode_cfg(
        src_base, dst_base, src_layout->cut_dim, dst_layout->cut_dim);
    dma_cmd.bank_cfg = 0U;
    dma_cmd.word15 = 0U;
    npuBinaryDataFromObject(&binary, dma_cmd);
    cmd->loadBinary(binary);
}

static inline void
dma_cmd_launch_move_layout(uint32_t device_id, uintptr_t src_base,
                           uintptr_t dst_base, const DmaLayout *src_layout,
                           const DmaLayout *dst_layout,
                           uint32_t sync_indicator,
                           uint32_t set_completion_sync)
{
    NpuCmd cmd;
    dma_cmd_init_move_layout(&cmd, device_id, src_base, dst_base,
                             src_layout, dst_layout, sync_indicator,
                             set_completion_sync);
    cmd.launchCmd();
}

static inline void
dma_cmd_launch_move_layout_stage(uint32_t device_id, uintptr_t src_base,
                                 uintptr_t dst_base,
                                 const DmaLayout *src_layout,
                                 const DmaLayout *dst_layout,
                                 uint32_t command_stage,
                                 uint32_t sync_indicator,
                                 uint32_t set_completion_sync)
{
    NpuCmd cmd;
    dma_cmd_init_move_layout_stage(
        &cmd, device_id, src_base, dst_base, src_layout, dst_layout,
        command_stage, sync_indicator, set_completion_sync);
    cmd.launchCmd();
}

static inline void
dma_cmd_launch_move_layout_at(uint64_t port_base, uint32_t device_id,
                              uintptr_t src_base, uintptr_t dst_base,
                              const DmaLayout *src_layout,
                              const DmaLayout *dst_layout,
                              uint32_t sync_indicator,
                              uint32_t set_completion_sync)
{
    NpuCmd cmd;
    dma_cmd_init_move_layout(&cmd, device_id, src_base, dst_base,
                             src_layout, dst_layout, sync_indicator,
                             set_completion_sync);
    cmd.launchCmdAt(port_base);
}

static inline void
dma_cmd_launch_move_layout_stage_at(uint64_t port_base, uint32_t device_id,
                                    uintptr_t src_base, uintptr_t dst_base,
                                    const DmaLayout *src_layout,
                                    const DmaLayout *dst_layout,
                                    uint32_t command_stage,
                                    uint32_t sync_indicator,
                                    uint32_t set_completion_sync)
{
    NpuCmd cmd;
    dma_cmd_init_move_layout_stage(
        &cmd, device_id, src_base, dst_base, src_layout, dst_layout,
        command_stage, sync_indicator, set_completion_sync);
    cmd.launchCmdAt(port_base);
}

#endif
