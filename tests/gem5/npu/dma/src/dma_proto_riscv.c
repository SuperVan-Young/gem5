#include <stdint.h>
#include <string.h>

#define CMDQ_BASE 0x70000000UL
#define CMD_BYTES 64UL
#define CTRL_ADDR (CMDQ_BASE + CMD_BYTES)

#define DEVICE_TYPE_SYNC 0x1U
#define DEVICE_TYPE_DMA 0x4U
#define DEVICE_ID 0x0U
#define SYNC_WAIT_OPCODE 0x0U

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

static inline void
mmio_write32(uint64_t addr, uint32_t value)
{
    *(volatile uint32_t *)addr = value;
}

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

static uint32_t
build_dma_header(uint32_t xfer_mode, uint32_t sync_idx)
{
    const uint32_t op = (0U << 5) | ((xfer_mode & 0x7U) << 2);
    return ((DEVICE_TYPE_DMA & 0xFU) << 28) |
           ((DEVICE_ID & 0xFU) << 24) |
           ((op & 0xFFU) << 16) |
           ((sync_idx & 0xFFU) << 8);
}

static uint32_t
build_sync_wait_header(uint32_t sync_idx)
{
    return ((DEVICE_TYPE_SYNC & 0xFU) << 28) |
           ((DEVICE_ID & 0xFU) << 24) |
           ((SYNC_WAIT_OPCODE & 0xFFU) << 16) |
           ((sync_idx & 0xFFU) << 8);
}

static void
push_words(const uint32_t words[16])
{
    for (unsigned i = 0; i < 16; ++i) {
        mmio_write32(CMDQ_BASE + (i * 4U), words[i]);
    }
    mmio_write32(CTRL_ADDR, 0);
}

static void
push_dma(uintptr_t src_base, uintptr_t dst_base,
         Layout src_layout, Layout dst_layout,
         uint32_t xfer_mode, uint32_t sync_idx)
{
    uint32_t words[16];
    for (unsigned i = 0; i < 16; ++i) {
        words[i] = 0;
    }

    words[15] = build_dma_header(xfer_mode, sync_idx);
    words[14] = (uint32_t)src_base;
    words[13] = (uint32_t)dst_base;
    words[12] = src_layout.h;
    words[11] = src_layout.w;
    words[10] = src_layout.c;
    words[9] = src_layout.stride_h;
    words[8] = src_layout.stride_w;
    words[7] = src_layout.stride_c;
    words[6] = dst_layout.stride_h;
    words[5] = dst_layout.stride_w;
    words[4] = dst_layout.stride_c;
    words[3] = ((uint32_t)dst_layout.k << 16) | src_layout.k;

    push_words(words);
}

static void
push_sync_wait(uint32_t sync_idx)
{
    uint32_t words[16];
    for (unsigned i = 0; i < 16; ++i) {
        words[i] = 0;
    }
    words[15] = build_sync_wait_header(sync_idx);
    push_words(words);
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
    push_dma(SRC_DRAM0, DST_SPM0, layout, layout, XFER_DRAM_TO_SPM, 5);
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_basic_spm_to_dram(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_DRAM0, layout.stride_h * layout.h);
    fill_tensor(SRC_SPM0, layout);
    push_dma(SRC_SPM0, DST_DRAM0, layout, layout, XFER_SPM_TO_DRAM, 6);
    return poll_until_match(DST_DRAM0, layout) ? 0 : 1;
}

static int
scenario_hwc_to_blocked(void)
{
    Layout src = make_layout(2, 4, 8, 0);
    Layout dst = make_layout(2, 4, 8, 2);
    clear_region(DST_SPM0, dst.stride_h * dst.h);
    fill_tensor(SRC_DRAM0, src);
    push_dma(SRC_DRAM0, DST_SPM0, src, dst, XFER_DRAM_TO_SPM, 7);
    return poll_until_match(DST_SPM0, dst) ? 0 : 1;
}

static int
scenario_blocked_to_blocked(void)
{
    Layout src = make_layout(2, 8, 4, 2);
    Layout dst = make_layout(2, 8, 4, 4);
    clear_region(DST_DRAM0, dst.stride_h * dst.h);
    fill_tensor(SRC_SPM0, src);
    push_dma(SRC_SPM0, DST_DRAM0, src, dst, XFER_SPM_TO_DRAM, 8);
    return poll_until_match(DST_DRAM0, dst) ? 0 : 1;
}

static int
scenario_buffer_size_forces_batching(void)
{
    Layout layout = make_layout(2, 8, 8, 0);
    clear_region(DST_SPM0, layout.stride_h * layout.h);
    fill_tensor(SRC_DRAM0, layout);
    push_dma(SRC_DRAM0, DST_SPM0, layout, layout, XFER_DRAM_TO_SPM, 9);
    return poll_until_match(DST_SPM0, layout) ? 0 : 1;
}

static int
scenario_sync_completion(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    clear_region(DST_SPM0, layout.stride_h * layout.h);
    clear_region(DST_DRAM1, layout.stride_h * layout.h);
    fill_tensor(SRC_DRAM0, layout);

    push_dma(SRC_DRAM0, DST_SPM0, layout, layout, XFER_DRAM_TO_SPM, 11);
    push_sync_wait(11);
    push_dma(DST_SPM0, DST_DRAM1, layout, layout, XFER_SPM_TO_DRAM, 12);

    return poll_until_match(DST_DRAM1, layout) ? 0 : 1;
}

static int
scenario_invalid_address(void)
{
    Layout layout = make_layout(2, 4, 8, 0);
    push_dma(0x10000000UL, DST_SPM0, layout, layout, XFER_DRAM_TO_SPM, 13);
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
    if (strcmp(argv[1], "invalid_address") == 0) {
        return scenario_invalid_address();
    }

    return 3;
}
