# VPU 旧式双设备基础测试

## 测试目的

验证 legacy VPU 双设备路径的基础执行行为是否稳定，包括不同 device id、不同读写 mask 和不同 repetition 下的命令执行结果与统计计数。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和两个 `VPU` 实例。
测试重点是双设备的基本命令接收、SPM 读写和阶段统计，不引入额外的系统级仲裁场景。

## 仿真程序

workload 依次向两个 VPU 发送四条 legacy 命令，覆盖两台设备上的不同同步标识、读写 mask 和 repetition。
程序会在完成后校验 SPM 的最终内容，并输出各设备的预期统计。

## 预期行为

仿真结束时，SPM 内容应与软件侧推导的结果一致，两个 VPU 的完成命令数和阶段统计应匹配预期，并输出稳定的通过标记 `VPU_LEGACY_DUAL_DEVICE_BASIC_PASS`。
