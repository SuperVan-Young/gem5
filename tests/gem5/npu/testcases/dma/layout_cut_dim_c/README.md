测试目的
验证 DMA 对 `cut_dim = C` 的布局语义解释是否正确，确认带 `k` 参数但沿通道维解释时，DMA 仍能按预期搬运同一个逻辑张量。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。源张量位于 DRAM，目标张量位于 SPM。

仿真程序
`workload.c` 构造一个逻辑形状为 `2 x 4 x 8` 的张量，并把源与目标都按 `k=2, cut_dim=C` 的地址规则解释。程序发射一条 `move_layout` 命令后，软件按通道维切分的解释方式逐元素核对结果。

预期行为
程序正常退出，并打印 `DMA_LAYOUT_CUT_DIM_C_PASS`。DMA 应只完成 1 条命令，目标张量在 `cut_dim=C` 解释下应与源张量逻辑值一致。
