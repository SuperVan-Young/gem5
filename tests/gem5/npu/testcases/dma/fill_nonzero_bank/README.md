# DMA fill_nonzero_bank

## 测试目的
验证 DMA 的 fill 模式可以把 DMA 内部 bank 1 的整个 64B 行写成固定非零字节值。这个 case 只测 bank fill 的非零路径。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。
测试只发射当前目录对应的那一类 DMA 宏命令，不和其他 fill、transpose 或 queue 场景混跑。

## 仿真程序
程序发射一条 fill 命令，目标空间选择 DMA bank，目标 bank 固定为 1，fill 值固定为 `0x5a`。`workload.c` 只选择 `scenario_fill_nonzero_bank()`。
共享头 `tests/gem5/npu/utils/dma_functional_cases.hh` 只保存可复用的数据准备和命令构造逻辑；当前目录的 `workload.c` 负责把入口固定到这一条场景。

## 预期行为
gem5 应输出 `DMA_BANK_FILL_OBSERVE bank=1 value=90 required=64 checksum=5760`，说明整行都被写成 `0x5a`。随后应看到 `DMA_SUMMARY scenario=fill_nonzero_bank ...` 和 `DMA_SCENARIO_PASS=fill_nonzero_bank`。
