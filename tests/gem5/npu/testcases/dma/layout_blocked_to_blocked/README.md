测试目的
验证 DMA 对两种 blocked 布局之间的重排能力：把源张量从 `k=2` blocked 布局转换成 `k=4` blocked 布局。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。源张量放在 SPM，目标张量写回 DRAM。

仿真程序
`workload.c` 构造一个逻辑形状为 `2 x 8 x 4` 的源张量，并按 `k=2` blocked 方式写入 SPM；目标区域按 `k=4` blocked 布局解释。程序发射一条 `move_layout` 命令做 blocked-to-blocked 转换，再逐元素验证转换后的逻辑值。

预期行为
程序正常退出，并打印 `DMA_LAYOUT_BLOCKED_TO_BLOCKED_PASS`。DMA 应只完成 1 条命令，目标区域按 `k=4` blocked 布局读取时应与源张量的逻辑值一致。
