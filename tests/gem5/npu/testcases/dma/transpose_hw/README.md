# DMA transpose_hw

## 测试目的
验证 DMA 的 transpose 模式可以交换张量的 H 和 W 两个维度，同时保持其余维度顺序不变。这个 case 只测 H<->W 交换。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。
测试只发射当前目录对应的那一类 DMA 宏命令，不和其他 fill、transpose 或 queue 场景混跑。

## 仿真程序
程序把源张量放在 DRAM，目标张量放在 SPM，发射一条 transpose 命令，把 `transpose_dim_a` 设为 H、`transpose_dim_b` 设为 W。`workload.c` 只调用 `scenario_transpose_hw()`。
共享头 `tests/gem5/npu/utils/dma_functional_cases.hh` 只保存可复用的数据准备和命令构造逻辑；当前目录的 `workload.c` 负责把入口固定到这一条场景。

## 预期行为
gem5 应输出 `DMA_TRANSPOSE_LATENCY dim_a=0 dim_b=1 ...`，并在 `DMA_SUMMARY scenario=transpose_hw ... iters=3 ...` 中体现三次内层迭代。随后应看到 `DMA_SCENARIO_PASS=transpose_hw`。
