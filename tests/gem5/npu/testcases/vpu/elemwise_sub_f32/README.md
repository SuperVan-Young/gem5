# VPU 浮点减法测试

## 测试目的

验证 VPU 的浮点减法路径是否稳定，覆盖基础二元浮点运算和结果写回。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
两个源向量分别放在独立端口，结果写回到独立目的端口。

## 仿真程序

workload 准备两个 4 元素的浮点向量，发射一条 `VSUB` 命令，并等待目的端口上的结果与 golden 接近。
程序不组合其它算子。

## 预期行为

仿真结束时，VPU 的完成计数应为 1，prologue/execute/epilogue/iteration 计数都应为 1，读响应计数应为 2，写响应计数应为 1。
配置脚本应输出稳定的 `VPU_ELEMWISE_SUB_F32_PASS` 标记。
