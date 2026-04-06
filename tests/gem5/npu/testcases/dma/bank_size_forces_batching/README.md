测试目的
验证 DMA 在 bank 容量不足以一次容纳整个张量时，会把一条逻辑搬运命令拆成多次内部迭代完成，而不是错误地一次性处理。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。与普通搬运 case 的区别是这里把 DMA `bank_size` 降到 32 字节，强制触发分批处理。

仿真程序
`workload.c` 构造一个逻辑形状为 `2 x 8 x 8` 的张量，并发射一条从 DRAM 到 SPM 的 `move_layout` 命令。由于 bank 容量只有 32 字节，这条命令不可能一次完成，程序最终仍按完整张量逐元素校验目标数据。

预期行为
程序正常退出，并打印 `DMA_BANK_SIZE_FORCES_BATCHING_PASS`。DMA 仍只应完成 1 条逻辑命令，但内部完成迭代数应为 4，用来证明该命令被拆成了多次处理。
