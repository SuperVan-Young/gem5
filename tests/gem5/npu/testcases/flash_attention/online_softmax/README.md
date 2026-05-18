# FlashAttention Online Softmax

## 测试目的

验证 FlashAttention Stage 5 的独立 online softmax helper，仅覆盖
`scores_block/m_prev/l_prev/m_next/l_next/p_block` 六类状态与输出，不引入
`O` 累积或最终 fused 集成。

## 仿真系统

使用 gem5 NPU testcase 默认的 RISC-V SE 模式最小系统，只运行静态软件
workload，不依赖 Stage 6 的端到端路径。

## 仿真程序

workload 构造固定 `scores` 样例并调用 `primitiveOnlineSoftmaxF32()`：

- 单块输入：从空状态开始，验证 `p_block` 与标准 softmax 对齐；
- 两块拼接输入：先处理前半块，再处理后半块，利用
  `m/l` 状态重建首块最终概率，并与整块标准 softmax 对齐；
- 同时检查每行 `m_next` 和 `l_next` 的数值契约。

## 预期行为

仿真通过时会打印：

- `FLASH_ATTENTION_ONLINE_SOFTMAX_SINGLE_BLOCK_PASS`
- `FLASH_ATTENTION_ONLINE_SOFTMAX_TWO_BLOCK_PASS`
- `FLASH_ATTENTION_ONLINE_SOFTMAX_PASS`
- `FLASH_ATTENTION_ONLINE_SOFTMAX_CONFIG_PASS`
