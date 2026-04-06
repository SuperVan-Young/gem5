# SEU issue queue 多端口隔离测试

## 测试目的

验证基础 `SEU` 的 mem issue queue 是否按 `MemoryPort` 隔离建模。落在同一个 mem issue queue 的宏指令应当串行，落在不同 mem issue queue 的宏指令应当可以并行推进。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个基础 `SEU`。
`SEU` 配置 2 个 `mem_side` 端口，因此默认会生成 2 个 mem issue queue。

## 仿真程序

workload 一次性发射 3 条 `load` 宏指令：

1. 第一条落在 port 0
2. 第二条也落在 port 0
3. 第三条落在 port 1

每条命令都带独立 sync indicator，所有 launch 完成后再统一等待完成。

## 预期行为

仿真结束时，`SEU` 应完成全部 3 条宏指令，`queue_occupancy` 为 0，`issue_busy` 为 false，`read_resps` 应等于全部 load uop 数之和，`write_resps` 应为 0。
同时 `max_active_micro_ops` 应至少为 2，说明不同 mem issue queue 的工作可以重叠执行，并输出稳定通过标记 `SEU_ISSUE_QUEUE_MEM_PORTS_PASS`。
