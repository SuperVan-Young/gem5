# SEU 基础阶段机测试

## 测试目的

验证基础 `SEU` 在最简单命令格式下的阶段机行为是否稳定，包括命令完成数、prologue/execute/epilogue 计数、iteration 计数和空闲状态。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个基础 `SEU`。
不启用 `SPM` 数据搬运路径，因此测试重点集中在 `SEU` 的命令接收与阶段推进。

## 仿真程序

workload 连续发射 5 条最小合法 VPU/SEU 命令。
每条命令的 `read_mask` 和 `write_mask` 都为 0，`repetition` 为 1，因此不会触发额外的读写响应。

## 预期行为

仿真结束时，退出原因应正确，`SEU` 队列占用应为 0，`issue_busy` 应为 false，且完成命令数、prologue/execute/epilogue/iteration 计数都应为 5。
`read_resps` 和 `write_resps` 都应为 0，并输出稳定的通过标记 `SEU_BASIC_PASS`。
