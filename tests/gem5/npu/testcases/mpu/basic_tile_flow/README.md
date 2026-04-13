# MPU basic_tile_flow 测试

## 测试目的

验证 `MPU` 在最小基础 tile 流程下能够完成 `mvin/load/compute/drain/mvout`
链路，并输出稳定的完成摘要。

## 仿真系统

该 testcase 复用当前 `MPU` 适配后的 gem5 NPU 测试系统配置，包含单 CPU、
`MegaCmdQueue`、`ScratchpadMemory` 和一个 `MPU` 实例。

## 仿真程序

workload 复用现有 `mpu_proto_riscv.c` 软件程序，通过 `basic_tile_flow`
scenario 发射最小合法 MPU 命令序列。

## 预期行为

仿真结束时应输出稳定的 `MPU_SUMMARY`，其中命令计数应为
`mvin=2 load=2 compute=1 drain=1 mvout=1`，并打印
`MPU_SCENARIO_PASS=basic_tile_flow`。
