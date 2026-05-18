## 测试目的

验证 FlashAttention Stage 6 的最终端到端集成 helper，完整覆盖
`Q/K/V` 量化、`QK^T`、`scores` 解量化与缩放、online softmax、`P` 量化、
`P*V` 与输出解量化，并要求最终 `O` 与 Stage 1 golden 对比通过。

## 仿真系统

复用 gem5 NPU testcase 默认的 RISC-V SE 模式最小系统，开启 MPU 相关
debug flag，以保留端到端 GEMM 命令统计用于 profiling。

## 仿真程序

workload 使用一个小尺寸但可精确验算的样例：

- `Q = [2, 2]`
- `K = [3, 2]`
- `V = [3, 2]`

程序先用 Stage 1 golden 生成 `scores/P/O`，再用 `flash_attention.hh`
中的 Stage 6 helper 跑完整硬件近似路径，并额外保留：

- `scores_i32`
- `scores_f32`
- `m_state/l_state`
- `P_f32`
- `out_i32`
- `out_f32`

这些关键中间结果的打印和校验，便于调试与 profiling。

## 预期行为

仿真通过时会打印：

- `FLASH_ATTENTION_BASIC_SCORES_I32=...`
- `FLASH_ATTENTION_BASIC_PROBS_F32=...`
- `FLASH_ATTENTION_BASIC_OUT_F32=...`
- `FLASH_ATTENTION_BASIC_F32_PASS`
- `MPU_EXIT_CODE=0`
- 对应端到端两次 tiled GEMM 的稳定 `MPU_SUMMARY`
