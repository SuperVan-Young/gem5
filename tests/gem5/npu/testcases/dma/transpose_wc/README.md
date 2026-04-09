# DMA transpose_wc

## 测试目的
验证 DMA 的 transpose 模式可以交换张量的 W 和 C 两个维度。这个 case 单独确认 W<->C 交换。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。
测试只发射当前目录对应的那一类 DMA 宏命令，不和其他 fill、transpose 或 queue 场景混跑。

## 仿真程序
程序把源张量放在 DRAM，目标张量放在另一段 DRAM，发射一条 transpose 命令，把 `transpose_dim_a` 设为 W、`transpose_dim_b` 设为 C。`workload.c` 只调用 `scenario_transpose_wc()`。
共享头 `tests/gem5/npu/utils/dma_functional_cases.hh` 只保存可复用的数据准备和命令构造逻辑；当前目录的 `workload.c` 负责把入口固定到这一条场景。

## 预期行为
gem5 应输出 `DMA_TRANSPOSE_LATENCY dim_a=1 dim_b=2 ...`，并最终输出 `DMA_SCENARIO_PASS=transpose_wc`。这说明 DMA 按 W<->C 的索引规则完成了重排。
