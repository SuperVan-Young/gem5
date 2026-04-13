# DMA reject_staged_transpose_interleaving

## 测试目的
验证 staged transpose sequence 活跃期间，不允许插入无关 DMA 宏命令破坏 sequence 一致性。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。程序只构造一个被故意打断的 staged transpose 场景。

## 仿真程序
程序先发一条 staged transpose `Load`，随后立刻插入一条普通 move_layout 命令，故意违反 active staged DMA sequence 的互斥规则。

## 预期行为
gem5 应 panic，并在 stderr 中匹配 `active staged DMA sequence does not allow interleaved DMA commands`。
