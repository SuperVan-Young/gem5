# VPU VADD Decode Smoke 测试

## 测试目的

验证 VPU 的 binary `VADD` 解码与最小向量加法执行路径是否稳定，重点检查写回结果是否正确。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
测试只覆盖 `VADD` 路径，不与其它算子混跑。

## 仿真程序

workload 在两个源 SPM slot 中放入简单向量，提交一条 `VADD` 命令，并检查目的 slot 的结果。
程序不串联其它 VPU 指令，也不引入额外同步场景。

## 预期行为

仿真结束时，向量加法结果应与期望一致，统计计数应全部为 1 或与两路读一次写的语义一致，并输出稳定通过标记 `VPU_DECODE_VADD_SMOKE_PASS`。
