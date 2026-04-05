# VPU unary exp 测试

## 测试目的

验证 `VEXP` 的数值结果是否正确，同时确认该命令走到 LUT 相关路径时的基础计数稳定。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、`MegaCmdQueue`、`SPM` 和一个 `VPU`。
测试只覆盖指数命令，不与其它 unary 功能混跑。

## 仿真程序

workload 发送一条 `VEXP` 命令，对 4 个浮点元素计算指数并写到目标槽位。

## 预期行为

仿真结束时，目标槽位应与 golden 结果一致，VPU 队列应为空，`issue_busy` 应为 false，基础计数应为 1，且 LUT 请求和命令计数应符合单条 `VEXP` 的行为。
测试通过时输出稳定的 `VPU_UNARY_EXP_PASS`。

