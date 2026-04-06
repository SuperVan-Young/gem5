# VPU softmax flat 测试

## 测试目的

验证 `VSOFTMAX` 的 flat 输入路径是否正确，重点检查结果值、LUT 路径一致性和基础计数。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、`MegaCmdQueue`、`SPM` 和一个 `VPU`。
测试只覆盖一种输入分布，不把多种 softmax 场景塞进同一个 testcase。

## 仿真程序

workload 发送一条 `VSOFTMAX` 命令，对 4 个相同浮点元素做 softmax 计算并写到目标槽位。

## 预期行为

仿真结束时，目标槽位应与 golden 结果一致，VPU 队列应为空，`issue_busy` 应为 false，基础计数应为 1，且 LUT 与 softmax 路径的计数/时序应一致。
测试通过时输出稳定的 `VPU_SOFTMAX_FLAT_PASS`。
