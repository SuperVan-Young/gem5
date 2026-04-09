测试目的
验证 DMA 从 SPM 回写到 DRAM 的基本搬运路径，确认源数据从片上存储搬到外部内存后仍保持逐元素一致。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。测试把源张量放在 SPM，把目标张量放在 DRAM。

仿真程序
`workload.c` 在 SPM 源区域生成一个 `2 x 4 x 8` 的字节张量，清空 DRAM 目标区域后发射一条 `move_layout` DMA 命令，把这块数据从 SPM 搬到 DRAM，随后软件逐元素核对目标内容。

预期行为
程序正常退出，并打印 `DMA_MOVE_SPM_TO_DRAM_BASIC_PASS`。DMA 统计应只有 1 条完成命令，队列占用为 0，DMA 不再 busy。
