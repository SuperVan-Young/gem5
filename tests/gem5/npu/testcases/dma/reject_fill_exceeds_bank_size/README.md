# DMA reject_fill_exceeds_bank_size

## 测试目的
验证 DMA 会拒绝违反下列约束的非法命令：fill 到 DMA bank 时，目标字节数不能超过 bank 容量。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。
测试只发射一条会命中当前拒绝规则的 DMA 宏命令，不和其他 reject 场景混跑。

## 仿真程序
程序构造一条 fill-bank 命令，并把目标张量大小设成 4097B。
当前 bank 容量是 4096B，DMA 无法在一个 bank 内容纳更大的目标区域。
当前目录的 `workload.c` 只负责选择这一条 reject 路径；共享头 `tests/gem5/npu/utils/dma_panic_cases.hh` 保存可复用的非法命令构造辅助，但不会把多个失败模式混在同一次运行里。

## 预期行为
gem5 应在 `DmaUnit` 检查该命令时触发 panic，stderr 中出现 `fill required_bytes=4097 exceeds bank_size=4096`。
如果命令意外没有触发 panic，测试壳子会主动报错，从而避免把“静默通过”误判为测试通过。
