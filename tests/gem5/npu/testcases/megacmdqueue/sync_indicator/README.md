# MegaCmdQueue Sync Indicator 测试

## 测试目的

验证 `MegaCmdQueue` 的 sync-indicator 交互路径是否稳定，包括等待命令入队、同步指示器置位，以及等待中的后续命令能否被正确释放。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个基础 `SEU`。
测试只需要 `MegaCmdQueue` 的同步指示器表和最基本的命令下游消费路径，不引入 `SPM`、`DMA` 或更复杂的系统拓扑。

## 仿真程序

workload 先发射一个 sync-wait 命令，让后续 VPU 命令等待指定的同步指示器。
随后发射一个带 `SNS` 置位的 VPU 宏命令，最后通过软件写同步指示器并用 `npu_cmd_sync_done()` 作为完成栅栏。

## 预期行为

仿真结束时，退出原因应正确，`MegaCmdQueue` 队列占用应为 0，`SEU` 队列占用也应为 0。
`config.py` 需要输出稳定的通过标记 `MEGACMDQUEUE_SYNC_INDICATOR_PASS`。
