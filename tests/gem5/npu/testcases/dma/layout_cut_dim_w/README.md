测试目的
验证 DMA 对 `cut_dim = W` 的 blocked 布局解释是否正确，也就是按宽度维度分块时，DMA 能否按同一逻辑张量正确读写数据。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。源张量位于 DRAM，目标张量位于 SPM。

仿真程序
`workload.c` 构造一个逻辑形状为 `2 x 8 x 4` 的张量，并把源与目标都按 `k=2, cut_dim=W` 的地址规则解释。程序发射一条 `move_layout` 命令后，软件按同样的宽度分块规则逐元素核对目标张量。

预期行为
程序正常退出，并打印 `DMA_LAYOUT_CUT_DIM_W_PASS`。DMA 应只完成 1 条命令，目标张量在 `cut_dim=W` 解释下应与源张量逻辑值一致。
