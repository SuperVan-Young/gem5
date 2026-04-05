# MegaCmdQueue 两次 Launch-Sync-Launch 测试

## 测试目的

验证 `MegaCmdQueue` 在连续两组“命令发射、sync wait、再次命令发射”序列下的同步释放行为是否稳定。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个基础 `SEU`。
测试只覆盖共享命令队列与同步指示器路径，不引入 `SPM`、`DMA` 或多核拓扑。

## 仿真程序

workload 先发射一个会设置同步指示器的命令，然后发射与该指示器关联的 sync wait；随后对另一组同步指示器重复同样的序列，最后用 `npu_cmd_sync_done()` 作为完成栅栏。

## 预期行为

仿真结束时，退出原因应正确，`MegaCmdQueue` 队列占用应为 0，`SEU` 队列占用应为 0，`SEU` 完成命令数应为 2，并输出 `MEGACMDQUEUE_LAUNCH_SYNC_LAUNCH_SYNC_PASS`。
