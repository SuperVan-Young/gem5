测试目的
验证 DMA 在搬运过程中执行布局重排的能力：把普通连续的 HWC 布局张量转换成带 `k=2` blocking 的目标布局。

仿真系统
采用 RISCV SE 模式单核系统，挂载一段普通 DRAM、一块 SPM、MegaCmdQueue 和 DMA 单元。源张量位于 DRAM，目标张量位于 SPM。

仿真程序
`workload.c` 在 DRAM 中构造一个逻辑形状为 `2 x 4 x 8` 的连续 HWC 张量，目标区域按 `k=2` blocked 布局解释。程序发射一条 `move_layout` 命令让 DMA 在复制的同时做布局变换，并逐元素按目标布局地址规则核对结果。

预期行为
程序正常退出，并打印 `DMA_LAYOUT_HWC_TO_BLOCKED_PASS`。DMA 应只完成 1 条命令，队列清空，软件按 blocked 布局读取目标数据时应看到与源张量一致的逻辑值。
