# MPU basic_gemm_single_tile 测试

## 测试目的

验证基于当前 `MPU` 命令流封装的基础 GEMM 工具可以正确完成单个 tile 的
`INT8 x INT8 -> INT32` 矩阵乘法。

## 仿真系统

该 testcase 复用当前 `MPU` 的 gem5 NPU 测试系统配置，包含单 CPU、
`MegaCmdQueue`、`ScratchpadMemory` 和一个 `MPU` 实例。

## 仿真程序

workload 使用新的 `mpu_gemm.hh` 与 `golden/mpu_gemm.hh`：
- 先在 SPM 中放置 row-major 的 A/B/C 矩阵
- 再通过 `mvin/load/compute/drain/mvout` 发射一个完整 tile episode
- 最后用 golden GEMM 结果检查 C 矩阵

## 预期行为

仿真结束时应输出稳定的 `MPU_SUMMARY`，命令计数应为
`mvin=2 load=2 compute=1 drain=1 mvout=1`，并打印
`MPU_SCENARIO_PASS=basic_gemm_single_tile`。
