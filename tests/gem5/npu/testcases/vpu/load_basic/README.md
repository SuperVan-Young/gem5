# VPU Load 基础测试

## 测试目的

验证 VPU 的 `load` 命令路径是否稳定，重点检查命令提交、同步完成和计数统计是否正确。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
测试只覆盖 `load` 路径，不再和 `store` 混合验证。

## 仿真程序

workload 先在两个 SPM 端口写入源数据，再提交一条 `load` 命令并等待同步完成。
程序不做额外的写回验证，测试重点放在 `load` 命令本身。

## 预期行为

仿真结束时，`load` 命令应成功完成，计数统计应符合单次命令语义，并输出稳定通过标记 `VPU_LOAD_BASIC_PASS`。
