# DMA transpose_overlap_hazard_gated

## 测试目的
验证 staged transpose 在遇到 destination-line hazard 时会 DMA-local 地抑制不安全 overlap，而不是为了追求 overlap 破坏功能正确性。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。测试只发射当前目录对应的 staged transpose 场景，不和其他 DMA case 混跑。

## 仿真程序
程序构造一个 `2x2x2` 的源 tensor，按 H<->W transpose 发射 staged `Load`、`Compute`、`Store` 三条命令。两个 iteration 会共享同一条目标 cache line，因此第二个 `Load` 必须等待前一个 `Store` 退休后才能继续。

## 预期行为
gem5 应看到 `DMA_SUMMARY scenario=transpose_overlap_hazard_gated cmds=3 ... overlap=0 ...`，随后输出 `DMA_SCENARIO_PASS=transpose_overlap_hazard_gated`。这说明 DMA 保留了功能正确性而没有强行制造 overlap。
