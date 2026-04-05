# VPU FMA 原位写回测试

## 测试目的

验证 VPU 的 `FMA` 原位写回路径是否稳定，重点检查结果覆盖到源端口的行为。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
三路输入分别放在三个源端口，结果写回到第一个源端口。

## 仿真程序

workload 准备三组 4 元素浮点向量，发射一条 `VFMA` 命令，并等待源端口上的结果与 golden 接近。
程序不叠加独立目的端口写回场景。

## 预期行为

仿真结束时，VPU 的完成计数应为 1，prologue/execute/epilogue/iteration 计数都应为 1，读响应计数应为 3，写响应计数应为 1。
配置脚本应输出稳定的 `VPU_FMA_INPLACE_PASS` 标记。
