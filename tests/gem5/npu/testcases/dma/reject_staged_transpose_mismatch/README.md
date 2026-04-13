# DMA reject_staged_transpose_mismatch

## 测试目的
验证 staged transpose 的 `Load/Compute/Store` 三条命令必须描述同一个 transpose descriptor，不能在 sequence 中途切换目标描述符。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。程序只构造一个 descriptor 被故意改坏的 staged transpose 场景。

## 仿真程序
程序先发一条合法 staged transpose `Load`，再发一条把目的地址改掉的 staged transpose `Compute`，故意让两条命令不再匹配同一个 active staged sequence。

## 预期行为
gem5 应 panic，并在 stderr 中匹配 `staged transpose command does not match the active staged sequence`。
