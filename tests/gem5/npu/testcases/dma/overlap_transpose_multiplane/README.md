# DMA overlap_transpose_multiplane

## 测试目的
验证 DMA 只通过自身 staged transpose 运行时设计，就能在多 iteration transpose 中观察到 batch-level overlap，同时保持功能正确。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。测试只发射当前目录对应的 staged transpose 场景，不和其他 DMA case 混跑。

## 仿真程序
程序构造一个 `8x8x2` 的源 tensor，按 H<->W transpose 发射 staged `Load`、`Compute`、`Store` 三条命令。两个 iteration 的目的 cache line 天然分离，既能观察 overlap，也不会把 destination-line hazard 混进这个验证里。case 先检查最终目标 tensor，再检查 DMA-local overlap 计数器。

## 预期行为
gem5 应看到 `DMA_SUMMARY scenario=overlap_transpose_multiplane cmds=3 ... overlap=<non-zero> ...`，随后输出 `DMA_SCENARIO_PASS=overlap_transpose_multiplane`。
