# VPU unary i2f 测试

## 测试目的

验证 `VCVT_I2F` 的整数转浮点路径是否正确，重点检查结果值和基础计数。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、`MegaCmdQueue`、`SPM` 和一个 `VPU`。
测试只覆盖一次类型转换，不混入其它 unary 行为。

## 仿真程序

workload 发送一条 `VCVT_I2F` 命令，把 4 个整数元素转换成浮点结果并写到目标槽位。

## 预期行为

仿真结束时，目标槽位应与 golden 结果一致，VPU 队列应为空，`issue_busy` 应为 false，且基础计数都应为 1。
测试通过时输出稳定的 `VPU_UNARY_I2F_PASS`。

