# VPU reduce max f32 测试

## 测试目的

验证 `VREDUCE_MAX` 的浮点路径是否正确，重点检查规约结果和基础计数。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、`MegaCmdQueue`、`SPM` 和一个 `VPU`。
测试只覆盖浮点最大值规约，不混入其它 reduce 行为。

## 仿真程序

workload 发送一条 `VREDUCE_MAX` 命令，对 4 个浮点元素做最大值规约并写到目标槽位。

## 预期行为

仿真结束时，目标槽位应与 golden 结果一致，VPU 队列应为空，`issue_busy` 应为 false，且基础计数都应为 1。
测试通过时输出稳定的 `VPU_REDUCE_MAX_F32_PASS`。

