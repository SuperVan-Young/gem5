# DMA staged_move_layout_basic

## 测试目的
验证 DMA 已经把 `move_layout` 的非 legacy stage surface 落到真实执行路径上。这个 case 只覆盖单 batch 的 `Load + Compute + Store` 三条 staged 命令，不混入 overlap。

## 仿真程序
程序准备一块 64B 的源 tensor，依次发射三条 staged `move_layout` 命令：`Load`、`Compute`、`Store`。最终只检查目标 tensor 的功能结果，不依赖调试输出。

## 预期行为
gem5 应看到 `DMA_SUMMARY scenario=staged_move_layout_basic cmds=3 ... overlap=0 ...`，随后输出 `DMA_SCENARIO_PASS=staged_move_layout_basic`。
