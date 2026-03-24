#include <stdint.h>
#include <string.h>

#include "npu_sync.hh"

#define DMA_DEVICE_ID 0x0U

#define XFER_DRAM_TO_SPM 0x0U
#define XFER_SPM_TO_DRAM 0x1U
#define XFER_SPM_TO_SPM 0x2U
#define XFER_DRAM_TO_DRAM 0x3U

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
} Layout;

static inline uintptr_t
coord_addr(uintptr_t base, Layout layout, uint32_t y, uint32_t x, uint32_t z)
{
    if (layout.k == 0) {
        return base + (uintptr_t)y * layout.stride_h +
               (uintptr_t)x * layout.stride_w +
               (uintptr_t)z * layout.stride_c;
    }

    return base + (uintptr_t)y * layout.stride_h +
           (uintptr_t)(x / layout.k) * layout.stride_c * layout.c +
           (uintptr_t)z * layout.stride_c +
           (uintptr_t)(x % layout.k) * layout.stride_w;
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

static void
build_dma_cmd_with_data_type(NpuCmd *cmd, uintptr_t src_base,
                             uintptr_t dst_base, Layout src_layout,
                             Layout dst_layout, uint32_t data_type,
                             uint32_t xfer_mode, uint32_t sync_idx,
                             uint32_t set_completion_sync)
{
    const uint32_t op =
        ((data_type & 0x7U) << 5) | ((xfer_mode & 0x7U) << 2);

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
}

static void
build_dma_cmd(NpuCmd *cmd, uintptr_t src_base, uintptr_t dst_base,
              Layout src_layout, Layout dst_layout, uint32_t xfer_mode,
              uint32_t sync_idx, uint32_t set_completion_sync)
{
    build_dma_cmd_with_data_type(cmd, src_base, dst_base, src_layout,
                                 dst_layout, 0U, xfer_mode, sync_idx,
                                 set_completion_sync);
}

static void
launch_dma(uintptr_t src_base, uintptr_t dst_base, Layout src_layout,
           Layout dst_layout, uint32_t xfer_mode, uint32_t sync_idx,
           uint32_t set_completion_sync)
{
    NpuCmd cmd;

    build_dma_cmd(&cmd, src_base, dst_base, src_layout, dst_layout,
                  xfer_mode, sync_idx, set_completion_sync);
    cmd.launchCmd();
}

static void
launch_dma_with_data_type(uintptr_t src_base, uintptr_t dst_base,
                          Layout src_layout, Layout dst_layout,
                          uint32_t data_type, uint32_t xfer_mode,
                          uint32_t sync_idx, uint32_t set_completion_sync)
{
    NpuCmd cmd;

    build_dma_cmd_with_data_type(&cmd, src_base, dst_base, src_layout,
                                 dst_layout, data_type, xfer_mode, sync_idx,
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
make_layout(uint32_t h, uint32_t w, uint32_t c, uint16_t k)
{
    Layout layout;
    layout.h = h;
    layout.w = w;
    layout.c = c;
    layout.k = k;
    if (k == 0) {
        layout.stride_c = 1;
        layout.stride_w = c;
        layout.stride_h = w * c;
    } else {
        layout.stride_w = 1;
        layout.stride_c = k;
        layout.stride_h = w * c;
    }
    return layout;
}

static int
scenario_basic_dram_to_spm(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, layout.stride_h * layout.h);
    fill_tensor(SRC_DRAM0, layout);
    launch_dma(SRC_DRAM0, DST_SPM0, layout, layout, XFER_DRAM_TO_SPM, 5, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_basic_spm_to_dram(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_DRAM0, layout.stride_h * layout.h);
    fill_tensor(SRC_SPM0, layout);
    launch_dma(SRC_SPM0, DST_DRAM0, layout, layout, XFER_SPM_TO_DRAM, 6, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_DRAM0, layout) ? 0 : 1;
}

static int
scenario_spm_to_spm(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, layout.stride_h * layout.h);
    fill_tensor(SRC_SPM0, layout);
    launch_dma(SRC_SPM0, DST_SPM0, layout, layout, XFER_SPM_TO_SPM, 10, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_dram_to_dram(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_DRAM1, layout.stride_h * layout.h);
    fill_tensor(SRC_DRAM0, layout);
    launch_dma(SRC_DRAM0, DST_DRAM1, layout, layout, XFER_DRAM_TO_DRAM,
               13, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_DRAM1, layout) ? 0 : 1;
}

static int
scenario_hwc_to_blocked(void)
{
    Layout src = make_layout(2, 4, 8, 0);
    Layout dst = make_layout(2, 4, 8, 2);
    clear_region(DST_SPM0, dst.stride_h * dst.h);
    fill_tensor(SRC_DRAM0, src);
    launch_dma(SRC_DRAM0, DST_SPM0, src, dst, XFER_DRAM_TO_SPM, 7, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, dst) ? 0 : 1;
}

static int
scenario_blocked_to_blocked(void)
{
    Layout src = make_layout(2, 8, 4, 2);
    Layout dst = make_layout(2, 8, 4, 4);
    clear_region(DST_DRAM0, dst.stride_h * dst.h);
    fill_tensor(SRC_SPM0, src);
    launch_dma(SRC_SPM0, DST_DRAM0, src, dst, XFER_SPM_TO_DRAM, 8, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_DRAM0, dst) ? 0 : 1;
}

static int
scenario_buffer_size_forces_batching(void)
{
    Layout layout = make_layout(2, 8, 8, 0);
    clear_region(DST_SPM0, layout.stride_h * layout.h);
    fill_tensor(SRC_DRAM0, layout);
    launch_dma(SRC_DRAM0, DST_SPM0, layout, layout, XFER_DRAM_TO_SPM, 9, 0);
    npu_cmd_sync_done();
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_sync_completion(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, layout.stride_h * layout.h);
    clear_region(DST_DRAM1, layout.stride_h * layout.h);
    fill_tensor(SRC_DRAM0, layout);

    launch_dma(SRC_DRAM0, DST_SPM0, layout, layout,
               XFER_DRAM_TO_SPM, 11, 1);
    npu_launch_sync_wait(DMA_DEVICE_ID, 11, 0, 0, 0);
    launch_dma(DST_SPM0, DST_DRAM1, layout, layout,
               XFER_SPM_TO_DRAM, 12, 0);
    npu_cmd_sync_done();

    return poll_until_match(DST_DRAM1, layout) ? 0 : 1;
}

static int
scenario_queued_chain(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, layout.stride_h * layout.h);
    clear_region(DST_DRAM1, layout.stride_h * layout.h);
    fill_tensor(SRC_DRAM0, layout);

    launch_dma(SRC_DRAM0, DST_SPM0, layout, layout,
               XFER_DRAM_TO_SPM, 21, 0);
    launch_dma(DST_SPM0, DST_DRAM1, layout, layout,
               XFER_SPM_TO_DRAM, 22, 0);
    npu_cmd_sync_done();

    return poll_until_match(DST_DRAM1, layout) ? 0 : 1;
}

static int
scenario_invalid_destination_address(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_dma(SRC_DRAM0, 0x10000000UL, layout, layout,
               XFER_DRAM_TO_SPM, 23, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_invalid_blocked_k(void)
{
    Layout src = make_layout(2, 4, 8, 3);
    Layout dst = make_layout(2, 4, 8, 0);
    launch_dma(SRC_DRAM0, DST_SPM0, src, dst, XFER_DRAM_TO_SPM, 24, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_unsupported_data_type(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_dma_with_data_type(SRC_DRAM0, DST_SPM0, layout, layout, 1U,
                              XFER_DRAM_TO_SPM, 25, 0);
    for (;;) {
        asm volatile("" ::: "memory");
    }
}

static int
scenario_invalid_address(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    launch_dma(0x10000000UL, DST_SPM0, layout, layout,
               XFER_DRAM_TO_SPM, 13, 0);
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
    if (strcmp(argv[1], "invalid_destination_address") == 0) {
        return scenario_invalid_destination_address();
    }
    if (strcmp(argv[1], "invalid_blocked_k") == 0) {
        return scenario_invalid_blocked_k();
    }
    if (strcmp(argv[1], "unsupported_data_type") == 0) {
        return scenario_unsupported_data_type();
    }
    if (strcmp(argv[1], "invalid_address") == 0) {
        return scenario_invalid_address();
    }

    return 3;
}
