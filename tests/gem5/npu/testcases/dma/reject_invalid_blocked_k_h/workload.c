/*
 * Copyright (c) 2026
 * All rights reserved.
 *
 * This workload launches exactly one illegal DMA command. The
 * DMA_PANIC_CASE name matches the violated rule in this directory, so the
 * file still states which single reject scenario is under test. The shared
 * helper only reuses command-construction boilerplate and dispatches that
 * one scenario.
 */

#define DMA_PANIC_CASE invalid_blocked_k_h
#include "../../../utils/dma_panic_cases.hh"
