# FlashAttention Golden Contract

## 测试目的

验证 FlashAttention Stage 1 的软件 golden 在 workload 侧可独立调用，并固定
`Q/K/V/O` shape 语义、`1 / sqrt(dim)` scaling 语义以及 `scores/P/O` 基准行为。

## 仿真系统

单核 RISCV timing CPU 加默认 DRAM，仅运行纯软件 workload，不依赖后续
FlashAttention 硬件实现。

## 仿真程序

使用一个小尺寸 cross-attention 样例：

- `Q = [2, 2]`
- `K = [3, 2]`
- `V = [3, 2]`

workload 会分别调用：

- `npu_golden_qk_scores_f32(...)`
- `npu_golden_attention_probs_f32(...)`
- `npu_golden_flashattention_f32(...)`

并把结果与手工推导的 `scores/P/O` 进行比对。

## 预期行为

仿真通过时会打印：

- `FLASH_ATTENTION_GOLDEN_SCORES=...`
- `FLASH_ATTENTION_GOLDEN_PROBS=...`
- `FLASH_ATTENTION_GOLDEN_OUTPUT=...`
- `FLASH_ATTENTION_GOLDEN_CONTRACT_PASS`
