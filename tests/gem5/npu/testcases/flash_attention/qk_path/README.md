## 测试目的

验证 FlashAttention Stage 4 的独立 `QK^T` 路径，覆盖
`Q_f32/K_f32 -> 量化 -> K^T 物化 -> mpu_gemm -> 解量化 -> 乘 1/sqrt(dim)`，
并证明现有 `mpu_gemm` 可直接复用于 `scores_i32` 生成。

## 仿真系统

使用 gem5 NPU testcase 默认的 RISC-V SE 模式最小系统，开启 MPU 相关
debug flag 以观测命令统计，不引入 online softmax 或最终 FlashAttention
集成逻辑。

## 仿真程序

程序构造一个 `seq_q=3, seq_k=5, dim=4` 的独立样例，调用
`flash_attention.hh` 中的 Stage 4 helper：
- 对 `Q_f32/K_f32` 做逐行对称量化；
- 显式物化 `K_t_i8`，把问题映射为现有 `mpu_gemm` 的
  `A=[seq_q, dim]`、`B=[dim, seq_k]`、`C=[seq_q, seq_k]`；
- 对 `scores_i32` 做逐元素解量化并乘 `1/sqrt(dim)`；
- 同时输出并比对 `scores_i32`、`scores_f32`、`golden_scores_f32`。

## 预期行为

workload 打印：
- `FLASH_ATTENTION_QK_SCORES_I32=...`
- `FLASH_ATTENTION_QK_SCORES_F32=...`
- `FLASH_ATTENTION_QK_GOLDEN_SCORES_F32=...`
- `FLASH_ATTENTION_QK_PATH_PASS`

配置脚本复用现有 `mpu_proto.py`，因此 gem5 日志中还应出现：
- `MPU_EXIT_CODE=0`
- 与 4 个 tiled GEMM 对应的稳定 `MPU_SUMMARY`
