# VPU 整数减法测试

## 测试目的

验证 VPU 的整数减法路径是否稳定，重点检查二元运算和结果写回。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
命令从两个源端口取数，并将差值写回到独立的目的端口。

## 仿真程序

workload 准备两个 4 元素的整数向量，发射一条 `VSUB` 命令，并等待目的端口上的结果与 golden 一致。
程序不串联其它 opcode，也不复用额外功能。

## 预期行为

仿真结束时，VPU 的完成计数应为 1，prologue/execute/epilogue/iteration 计数都应为 1，读响应计数应为 2，写响应计数应为 1。
配置脚本应输出稳定的 `VPU_ELEMWISE_SUB_I32_PASS` 标记。
