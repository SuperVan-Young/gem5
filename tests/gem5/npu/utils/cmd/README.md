# Command Views

> 说明：本目录会随着 `utils/` 一起迁移到 `software_utils/cmd/`。迁移后，workload 侧的 `#include "cmd/common.hh"`、`#include "cmd/vpu.hh"`、`#include "cmd/dma.hh"` 形式保持不变，只切换 Makefile 的 include 搜索路径。

`common.hh` is the base command helper for NPU macro instructions.
It defines the shared macro-command header, low-level bit access helpers, and launch helpers.

Add extra headers in this directory only when a command view has stabilized enough that dedicated `getX()` / `setX()` helpers reduce testcase complexity.

Current specialized helpers:

- `vpu.hh`: wraps the stabilized VPU command layout used by VPU unit and
  system-level tests.
- `dma.hh`: wraps the merged DMA move-layout command format used by the current
  `DmaUnit` implementation. It centralizes DMA word placement, memory-space
  encoding, and completion-sync bit setup so system tests do not duplicate stale
  DMA packet layouts.
