# DMA overlap_move_layout_multibatch

## 测试目的
验证 DMA 只通过自身 staged `move_layout` 运行时设计，就能在多 batch 搬运中观察到 `Store(N)` 与 `Load(N+1)` 的 batch-level overlap，同时保持功能正确。

## 仿真程序
程序构造一个会因 `bank_size=64` 被切成多个 batch 的 tensor，依次发射 staged `Load`、`Compute`、`Store` 三条命令。每个 batch 对应不同的目标 cache line，既能观察 batch-level overlap，也不会把 line hazard 混进这个验证里。case 先检查最终目标 tensor，再检查 DMA-local overlap 计数器。

## 预期行为
gem5 应看到 `DMA_SUMMARY scenario=overlap_move_layout_multibatch cmds=3 ... overlap=<non-zero> ...`，随后输出 `DMA_SCENARIO_PASS=overlap_move_layout_multibatch`。
