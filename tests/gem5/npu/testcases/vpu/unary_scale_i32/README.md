# VPU unary scale i32 测试

## 测试目的

验证 `VSCALE` 的整数路径是否正常工作，重点检查原位写回、结果值和 VPU 基础计数是否稳定。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、`MegaCmdQueue`、`SPM` 和一个 `VPU`。
测试只启用一个 VPU 设备，不引入额外的系统级同步或第二个计算单元。

## 仿真程序

workload 发送一条 `VSCALE` 整数命令，对 4 个元素执行乘 3 运算，并把结果写回同一个 SPM 槽位。

## 预期行为

仿真结束时，源槽位应被正确更新，VPU 队列应为空，`issue_busy` 应为 false，且各项基础计数都应为 1。
测试通过时输出稳定的 `VPU_UNARY_SCALE_I32_PASS`。
