# DMA reject_staged_transpose_aliasing

## 测试目的
验证 staged transpose 的初始实现对外部 source/destination cache-line alias 采取保守拒绝，而不是在当前 DMA-local overlap 设计里冒险支持未证明安全的别名场景。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。程序只构造一个 source/destination 明显重叠的 staged transpose 场景。

## 仿真程序
程序发一条 staged transpose `Load`，让 source base 和 destination base 指向同一块外部地址范围，故意让 transpose 的 source line 与 destination line 落到同一 cache line。

## 预期行为
gem5 应 panic，并在 stderr 中匹配 `staged transpose requires source/destination cache-line disjointness`。
