# DMA staged_transpose_basic

## 测试目的
验证 DMA 已经把 transpose 的非 legacy stage surface 落到真实执行路径上。这个 case 只覆盖单 iteration 的 `Load + Compute + Store` 三条 staged transpose 命令，不混入 overlap。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。测试只发射当前目录对应的 staged transpose 场景，不和其他 DMA case 混跑。

## 仿真程序
程序准备一块 2x4x1 的源 tensor，目标 tensor 放在 SPM，并依次发射三条 staged transpose 命令：`Load`、`Compute`、`Store`。最终只检查目标 tensor 的功能结果，不依赖调试输出。

## 预期行为
gem5 应看到 `DMA_SUMMARY scenario=staged_transpose_basic cmds=3 reads=2 writes=1 overlap=0 ...`，随后输出 `DMA_SCENARIO_PASS=staged_transpose_basic`。
