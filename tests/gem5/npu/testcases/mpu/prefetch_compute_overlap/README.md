# MPU prefetch_compute_overlap 测试

## 测试目的

验证当前阶段的 `MPU` overlap 实现能够在不修改公共命令格式的前提下，让
下一块 tile 的 `mvin` 与当前 tile 的执行流水线发生重叠。

## 仿真系统

该 testcase 使用单 CPU、`MegaCmdQueue`、`ScratchpadMemory` 和一个
`MPU` 实例。`MPU` 保持主线 `SEU` 回调模型，不引入 DMA 或 VPU 参考行为。

## 仿真程序

workload 连续发射两组 `mvin/load/compute/drain/mvout` 命令，并使用交替
的 A/B/C buffer 索引，以便下一组 tile 的输入预取与当前组 tile 的执行
阶段发生安全重叠。

## 预期行为

仿真结束时，两组输出矩阵都应正确，`MPU_SUMMARY` 中应出现
`cmds=14`、`mvin=4 load=4 compute=2 drain=2 mvout=2`，并且
`max_active_uops` 应大于等于 2，证明出现了可观察的 overlap。
