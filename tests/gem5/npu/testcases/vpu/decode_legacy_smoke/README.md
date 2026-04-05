# VPU Legacy Decode Smoke 测试

## 测试目的

验证 VPU 的 legacy exec 解码与最小命令执行路径是否稳定，重点检查同步命令完成后 SPM 中的标量回写结果。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
测试只覆盖最基础的 legacy decode 路径，不引入额外的算子类型。

## 仿真程序

workload 仅提交一条 legacy exec 命令，并在单个 SPM slot 上检查执行结果。
程序不串联其它 VPU 算子，也不混入 load/store 之外的功能。

## 预期行为

仿真结束时，legacy exec 应正确完成，SPM 标量值应与期望一致，统计计数应全部为 1，并输出稳定通过标记 `VPU_DECODE_LEGACY_SMOKE_PASS`。
