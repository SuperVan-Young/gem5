# VPU FMA 基础写回测试

## 测试目的

验证 VPU 的 `FMA` 基础路径是否稳定，覆盖三输入计算和写回到独立目的端口。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
三路输入分别放在三个源端口，结果写回到第四个端口。

## 仿真程序

workload 准备三组 4 元素浮点向量，发射一条 `VFMA` 命令，并等待目的端口上的结果与 golden 接近。
程序不叠加原位写回场景。

## 预期行为

仿真结束时，VPU 的完成计数应为 1，prologue/execute/epilogue/iteration 计数都应为 1，读响应计数应为 3，写响应计数应为 1。
配置脚本应输出稳定的 `VPU_FMA_BASIC_PASS` 标记。
