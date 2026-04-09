测试目的
验证 DMA 在 SPM 内部不同地址窗口之间做原样搬运时，能够正确完成片上到片上的复制。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。本测试的源和目标都位于 SPM，只是地址窗口不同。

仿真程序
`workload.c` 在 SPM 源区域生成一个 `2 x 4 x 8` 的字节张量，清空另一段 SPM 目标区域后，发射一条 `move_layout` DMA 命令把数据在 SPM 内部搬运，之后逐元素核对目标区域。

预期行为
程序正常退出，并打印 `DMA_MOVE_SPM_TO_SPM_BASIC_PASS`。DMA 应只完成 1 条命令，队列清空，DMA 不再 busy。
