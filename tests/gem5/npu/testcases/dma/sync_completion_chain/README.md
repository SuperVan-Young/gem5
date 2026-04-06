# DMA sync_completion_chain

## 测试目的
验证 DMA 可以把“前一条命令完成”作为后一条命令的同步前提，形成一条最小 completion chain。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。
测试只发射当前目录对应的那一类 DMA 宏命令，不和其他 fill、transpose 或 queue 场景混跑。

## 仿真程序
程序先发一条 DRAM->SPM 的 move-layout 命令，并要求 DMA 在完成后设置 sync indicator；随后发出等待该 indicator 的 sync；最后再发一条 SPM->DRAM 的 move-layout 命令。`workload.c` 只调用 `scenario_sync_completion()`。
共享头 `tests/gem5/npu/utils/dma_functional_cases.hh` 只保存可复用的数据准备和命令构造逻辑；当前目录的 `workload.c` 负责把入口固定到这一条场景。

## 预期行为
gem5 最终应输出 `DMA_SCENARIO_PASS=sync_completion`。这说明第二次搬运确实发生在第一次 DMA 完成之后，而不是与其乱序交错。
