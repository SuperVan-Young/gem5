# Tile 级 MegaCmdQueue + SEU 测试

## 测试目的

验证最小 tile 级命令流中，`MegaCmdQueue` 到 `SEU` 的基本联通性和完成路径是否稳定。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `SEU`。
`SEU` 采用较长的调试处理延迟，用于让队列和执行单元在仿真中呈现更明显的流动过程。

## 仿真程序

workload 连续发射 6 条最小合法的 VPU/SEU 执行命令，随后执行一段固定自旋，让硬件模型有足够时间消费队列。

## 预期行为

仿真结束时，退出原因应正确，`MegaCmdQueue` 队列占用应为 0，`SEU` 队列占用应为 0，`SEU` 不应忙碌，完成命令数应不少于 6，并输出 `TILE_MEGACMDQUEUE_SEU_PASS`。
