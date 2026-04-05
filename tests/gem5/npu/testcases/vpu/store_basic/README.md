# VPU Store 基础测试

## 测试目的

验证 VPU 的 `store` 命令路径是否稳定，重点检查回写结果是否能正确恢复到 SPM。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
测试将 `load` 仅作为前置准备步骤，实际断言集中在 `store` 回写路径。

## 仿真程序

workload 先执行一次 `load` 以建立待回写的内部状态，然后污染 SPM，再提交一条 `store` 命令并检查回写结果。
程序不会把 `load` 和 `store` 作为同一个综合场景来判断，而是把 `store` 作为唯一关注点。

## 预期行为

仿真结束时，`store` 回写应恢复两个端口上的原始向量，计数统计应符合两条命令的语义，并输出稳定通过标记 `VPU_STORE_BASIC_PASS`。
