# SEU issue queue 混合资源测试

## 测试目的

验证基础 `SEU` 在 `exec`、`load`、`store` 三类宏指令混合发射时，能够按 issue queue 维度并行推进，不会因为单一全局 active command 模型而阻塞。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个基础 `SEU`。
`SEU` 配置 2 个 `mem_side` 端口，因此同时存在 1 个 exec issue queue 和 2 个 mem issue queue。

## 仿真程序

workload 一次性发射 3 条命令：

1. 一条 `exec` 命令
2. 一条落在 port 0 的 `load` 命令
3. 一条落在 port 1 的 `store` 命令

所有命令都在发射完成后统一等待 sync indicator。

## 预期行为

仿真结束时，`SEU` 应完成全部 3 条宏指令，`read_resps` 和 `write_resps` 分别与 load/store uop 数匹配，`executes` 与 exec uop 数匹配。
同时 `max_active_micro_ops` 应至少为 3，说明 exec 与两个 mem issue queue 的工作可以重叠进行，并输出稳定通过标记 `SEU_ISSUE_QUEUE_MIXED_PASS`。
