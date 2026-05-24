# FlashAttention FastOnlineSoftmax

## 测试目的

验证 `fastOnlineSoftmax` 的第一版 `VPU` 实现是否能在 `256 x 1024` 的
`score` block 上正确完成：

- 沿最后一个维度做 block softmax
- 根据输入的逐 query `m_prev`、`l_prev` 更新 `m_next`、`l_next`
- 输出当前 block 的 `p_block`

同时导出 profiling 结果，检查命令模板数量和实际执行的 `VPU` opcode 计数。

## 仿真系统

使用 gem5 NPU testcase 默认的 RISC-V SE 最小系统，打开：

- `SPM`
- `MegaCmdQueue`
- 单个 `VPU`

不引入 `MPU`、`DMA` 和端到端 FlashAttention 融合逻辑。

## 仿真程序

`workload.c` 会：

- 生成一个固定的 `256 x 1024` `scores_block`
- 生成固定的 `m_prev[256]` 与 `l_prev[256]`
- 用本地 CPU reference 计算期望的 `m_next`、`l_next`、`p_block`
- 调用 `fastOnlineSoftmax.hh` 中的 `VPU` primitive
- 从 `SPM` 读回结果并校验精度
- 打印稳定的性能和精度摘要

## 预期行为

仿真通过时应输出：

- `FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_SCENARIO=flash_attention_fast_online_softmax_256x1024`
- `FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_CMD ... status=PASS`
- `FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_ACCURACY ... status=PASS`
- `FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PROFILE ... status=PASS`
- `FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PASS`
- `FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_CONFIG_PASS`
