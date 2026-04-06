/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cmd/dma.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"

#ifndef CASE_NAME_STR
#error CASE_NAME_STR must be defined by the Makefile
#endif

#ifndef CASE_PASS_MARKER
#error CASE_PASS_MARKER must be defined by the Makefile
#endif

enum
{
    DMA_DEVICE_ID = 0U,
    MAX_POLL_ITERS = 2000000UL,
};

#define DRAM_BASE 0x20000000UL
#define SPM_BASE 0x60000000UL
#define SRC_DRAM0 (DRAM_BASE + 0x1000UL)
#define DST_DRAM0 (DRAM_BASE + 0x3000UL)
#define DST_DRAM1 (DRAM_BASE + 0x5000UL)
#define SRC_SPM0 (SPM_BASE + 0x1000UL)
#define DST_SPM0 (SPM_BASE + 0x3000UL)

static uintptr_t
coord_addr(uintptr_t base, DmaLayout layout, uint32_t y, uint32_t x,
           uint32_t z)
{
    const uintptr_t linear = base + (uintptr_t)y * layout.stride_h +
                             (uintptr_t)x * layout.stride_w +
                             (uintptr_t)z * layout.stride_c;

    if (layout.k == 0U) {
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

static uint8_t
pattern(uint32_t y, uint32_t x, uint32_t z)
{
    return (uint8_t)((y * 37U + x * 11U + z) & 0xffU);
}

static void
fill_tensor(uintptr_t base, DmaLayout layout)
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
    npu_mem_clear_bytes((volatile uint8_t *)base, (uint32_t)bytes);
}

static size_t
tensor_bytes(DmaLayout layout)
{
    return (size_t)layout.h * (size_t)layout.w * (size_t)layout.c;
}

static int
verify_tensor(uintptr_t base, DmaLayout layout)
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

static int
poll_until_match(uintptr_t base, DmaLayout layout)
{
    for (unsigned long iter = 0UL; iter < MAX_POLL_ITERS; ++iter) {
        if (verify_tensor(base, layout)) {
            return 1;
        }
        asm volatile("" ::: "memory");
    }
    return 0;
}

static DmaLayout
make_layout_with_cut(uint32_t h, uint32_t w, uint32_t c, uint16_t k,
                     uint32_t cut_dim)
{
    DmaLayout layout;
    layout.h = h;
    layout.w = w;
    layout.c = c;
    layout.k = k;
    layout.cut_dim = cut_dim;

    if (k == 0U || cut_dim == DMA_CUT_DIM_C) {
        layout.stride_c = 1U;
        layout.stride_w = c;
        layout.stride_h = w * c;
        return layout;
    }

    if (cut_dim == DMA_CUT_DIM_H) {
        layout.stride_h = 1U;
        layout.stride_c = k;
        layout.stride_w = c * k;
        return layout;
    }

    layout.stride_w = 1U;
    layout.stride_c = k;
    layout.stride_h = w * c;
    return layout;
}

static DmaLayout
make_layout(uint32_t h, uint32_t w, uint32_t c, uint16_t k)
{
    return make_layout_with_cut(h, w, c, k, DMA_CUT_DIM_W);
}

static int
run_move_case(uintptr_t src_base, uintptr_t dst_base, DmaLayout src_layout,
              DmaLayout dst_layout, uint32_t sync_indicator)
{
    clear_region(dst_base, tensor_bytes(dst_layout));
    fill_tensor(src_base, src_layout);

    dma_cmd_launch_move_layout(DMA_DEVICE_ID, src_base, dst_base, &src_layout,
                               &dst_layout, sync_indicator, 0U);
    npu_cmd_sync_done();
    return poll_until_match(dst_base, dst_layout) ? 0 : 1;
}

static int
run_case(const char *case_name)
{
    if (strcmp(case_name, "move_dram_to_spm_basic") == 0) {
        const DmaLayout layout = make_layout(2U, 4U, 8U, 0U);
        return run_move_case(SRC_DRAM0, DST_SPM0, layout, layout, 5U);
    }

    if (strcmp(case_name, "move_spm_to_dram_basic") == 0) {
        const DmaLayout layout = make_layout(2U, 4U, 8U, 0U);
        return run_move_case(SRC_SPM0, DST_DRAM0, layout, layout, 6U);
    }

    if (strcmp(case_name, "move_spm_to_spm_basic") == 0) {
        const DmaLayout layout = make_layout(2U, 4U, 8U, 0U);
        return run_move_case(SRC_SPM0, DST_SPM0, layout, layout, 10U);
    }

    if (strcmp(case_name, "move_dram_to_dram_basic") == 0) {
        const DmaLayout layout = make_layout(2U, 4U, 8U, 0U);
        return run_move_case(SRC_DRAM0, DST_DRAM1, layout, layout, 13U);
    }

    if (strcmp(case_name, "layout_hwc_to_blocked") == 0) {
        const DmaLayout src = make_layout(2U, 4U, 8U, 0U);
        const DmaLayout dst = make_layout(2U, 4U, 8U, 2U);
        return run_move_case(SRC_DRAM0, DST_SPM0, src, dst, 7U);
    }

    if (strcmp(case_name, "layout_blocked_to_blocked") == 0) {
        const DmaLayout src = make_layout(2U, 8U, 4U, 2U);
        const DmaLayout dst = make_layout(2U, 8U, 4U, 4U);
        return run_move_case(SRC_SPM0, DST_DRAM0, src, dst, 8U);
    }

    if (strcmp(case_name, "layout_cut_dim_h") == 0) {
        const DmaLayout layout =
            make_layout_with_cut(4U, 3U, 2U, 2U, DMA_CUT_DIM_H);
        return run_move_case(SRC_DRAM0, DST_SPM0, layout, layout, 31U);
    }

    if (strcmp(case_name, "layout_cut_dim_w") == 0) {
        const DmaLayout layout =
            make_layout_with_cut(2U, 8U, 4U, 2U, DMA_CUT_DIM_W);
        return run_move_case(SRC_DRAM0, DST_SPM0, layout, layout, 32U);
    }

    if (strcmp(case_name, "layout_cut_dim_c") == 0) {
        const DmaLayout layout =
            make_layout_with_cut(2U, 4U, 8U, 2U, DMA_CUT_DIM_C);
        return run_move_case(SRC_DRAM0, DST_SPM0, layout, layout, 33U);
    }

    if (strcmp(case_name, "bank_size_forces_batching") == 0) {
        const DmaLayout layout = make_layout(2U, 8U, 8U, 0U);
        return run_move_case(SRC_DRAM0, DST_SPM0, layout, layout, 9U);
    }

    return 3;
}

int
main(void)
{
    const int rc = run_case(CASE_NAME_STR);

    if (rc == 0) {
        printf("%s\n", CASE_PASS_MARKER);
    }

    return rc;
}
