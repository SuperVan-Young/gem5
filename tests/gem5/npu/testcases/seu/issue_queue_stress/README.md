# SEU issue queue 压力测试

## 测试目的

验证基础 `SEU` 在复杂命令流下不会卡死、不会丢命令，也不会因为 `dispatchQueue` 与多个 issue queue 并行推进而出现统计错乱。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个基础 `SEU`。
`SEU` 配置 2 个 `mem_side` 端口，且 `cmd_queue_depth` 被收紧到 4，以覆盖高于队列深度的命令流。

## 仿真程序

workload 一次性发射 10 条混合命令，覆盖：

1. `exec`
2. `load`
3. `store`
4. 同 port 串行
5. 跨 port 并行

所有命令都在发射完成后统一等待 sync indicator。

## 预期行为

仿真结束时，`SEU` 应完成全部 10 条宏指令，`queue_occupancy` 为 0，`issue_busy` 为 false，且 `read_resps`、`write_resps`、`executes` 与 workload 约定值一致。
`max_active_micro_ops` 应至少为 3，并输出稳定通过标记 `SEU_ISSUE_QUEUE_STRESS_PASS`。
