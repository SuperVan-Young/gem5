测试目的
验证 DMA 最基本的跨存储空间搬运能力：把一块位于 DRAM 的连续张量原样搬到 SPM，并确认数据内容没有被改写。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。测试程序把源张量放在 DRAM 地址窗口，把目标张量放在 SPM 地址窗口。

仿真程序
`workload.c` 先在 DRAM 源区域写入一个 `2 x 4 x 8` 的字节张量，再清空 SPM 目标区域，然后发射一条 `move_layout` DMA 命令，把这块张量从 DRAM 搬到 SPM，最后逐元素校验目标区域是否与源模式一致。

预期行为
程序正常退出，并打印 `DMA_MOVE_DRAM_TO_SPM_BASIC_PASS`。DMA 统计应表现为只完成 1 条命令，队列清空，单元不再 busy。
