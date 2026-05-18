## 测试目的

验证 FlashAttention Stage 6 子阶段 A 的独立 `PV` 路径，只覆盖
`P_f32/V_f32 -> 量化 -> P*V GEMM -> 解量化 -> O_f32`，不接入端到端
FlashAttention。

## 仿真系统

复用 gem5 NPU testcase 默认的 RISC-V SE 模式最小系统，开启 MPU 相关
debug flag 以观测命令统计。

## 仿真程序

程序构造一个 `seq_q=3, seq_k=5, dim_v=4` 的独立样例：
- `P_f32` 使用逐行量化，保持 FlashAttention 中 `P=[seq_q, seq_k]`
  的自然布局；
- `V_f32` 使用逐列量化，但量化后的 `V_i8` 仍然按
  `B=[seq_k, dim_v]` 的 dense row-major 形式存放，因此可直接复用现有
  `mpu_gemm`，无需 transpose 或 kernel 改动；
- 使用 `mpu_gemm` 生成 `out_i32=[seq_q, dim_v]`；
- 按 `P` 的行 scale 与 `V` 的列 scale 解量化为 `out_f32`；
- 同时输出并比对 `out_i32`、`out_f32` 与 golden `O`。

## 预期行为

workload 打印：
- `FLASH_ATTENTION_PV_OUT_I32=...`
- `FLASH_ATTENTION_PV_OUT_F32=...`
- `FLASH_ATTENTION_PV_GOLDEN_O_F32=...`
- `FLASH_ATTENTION_PV_PATH_PASS`

配置脚本复用现有 `mpu_proto.py`，因此 gem5 日志中还应出现：
- `MPU_EXIT_CODE=0`
- 与 4 个 tiled GEMM 对应的稳定 `MPU_SUMMARY`
