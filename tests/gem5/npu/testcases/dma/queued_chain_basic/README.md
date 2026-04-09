# DMA queued_chain_basic

## 测试目的
验证 DMA 命令队列可以连续接收两条同类命令，并按顺序把第一条的结果继续作为第二条的输入。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。
测试只发射当前目录对应的那一类 DMA 宏命令，不和其他 fill、transpose 或 queue 场景混跑。

## 仿真程序
程序连续发两条 move-layout 命令：第一条把数据从 DRAM 搬到 SPM，第二条再把同一批数据从 SPM 搬回另一段 DRAM。`workload.c` 只调用 `scenario_queued_chain()`。
共享头 `tests/gem5/npu/utils/dma_functional_cases.hh` 只保存可复用的数据准备和命令构造逻辑；当前目录的 `workload.c` 负责把入口固定到这一条场景。

## 预期行为
gem5 应看到 `DMA_SUMMARY scenario=queued_chain ... queue=0 cmdq=0 busy=0`，说明链式场景最终完成且队列排空。随后应看到 `DMA_SCENARIO_PASS=queued_chain`。
