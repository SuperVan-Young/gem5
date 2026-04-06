测试目的
验证 DMA 在 DRAM 内部不同地址区间之间做原样搬运时，能够正确完成外部内存到外部内存的复制。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。本测试的源和目标都位于 DRAM，只是地址区间不同。

仿真程序
`workload.c` 在 DRAM 源区域生成一个 `2 x 4 x 8` 的字节张量，清空另一段 DRAM 目标区域后发射一条 `move_layout` DMA 命令，随后逐元素检查目标区域是否已经得到同样的数据。

预期行为
程序正常退出，并打印 `DMA_MOVE_DRAM_TO_DRAM_BASIC_PASS`。DMA 应完成 1 条命令，队列占用归零，DMA 不再 busy。
