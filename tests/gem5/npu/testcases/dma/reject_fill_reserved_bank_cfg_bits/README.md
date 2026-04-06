# DMA reject_fill_reserved_bank_cfg_bits

## 测试目的
验证 DMA 会拒绝违反下列约束的非法命令：fill 的 `bank_cfg` 只能使用已定义的目标 bank 位。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。
测试只发射一条会命中当前拒绝规则的 DMA 宏命令，不和其他 reject 场景混跑。

## 仿真程序
程序先构造一条合法 fill-bank 命令，再把 `bank_cfg` 的保留位设成 1。
DMA 无法把含保留位的 `bank_cfg` 解释成合法 fill 目标，因此必须拒绝。
当前目录的 `workload.c` 只负责选择这一条 reject 路径；共享头 `tests/gem5/npu/utils/dma_panic_cases.hh` 保存可复用的非法命令构造辅助，但不会把多个失败模式混在同一次运行里。

## 预期行为
gem5 应在 `DmaUnit` 检查该命令时触发 panic，stderr 中出现 `reserved bank_cfg bits set for fill`。
如果命令意外没有触发 panic，测试壳子会主动报错，从而避免把“静默通过”误判为测试通过。
