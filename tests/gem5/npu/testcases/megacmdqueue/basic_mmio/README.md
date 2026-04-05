# MegaCmdQueue 基础 MMIO 测试

## 测试目的

验证 `MegaCmdQueue` 的基础 MMIO 行为是否稳定，包括命令写入、显式 `pop` 以及队列占用计数变化。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存和一个 `MegaCmdQueue`。
队列深度设置为 2，并且不启用同步指示器逻辑，因此测试重点只放在队列控制面行为。

## 仿真程序

workload 先连续提交两个宏命令，然后对队列控制寄存器执行一次显式 `pop`，最后再提交一个宏命令。
程序本身不引入 `SEU`、`SPM` 或 `DMA`，只依赖 `MegaCmdQueue` 的基本命令入队与出队路径。

## 预期行为

仿真结束时，`MegaCmdQueue` 的最终队列占用应为 2，且 `config.py` 需要打印稳定的 PASS 标记。
当前 testcase 的通过条件是退出原因正确、队列占用正确，并输出 `MEGACMDQUEUE_BASIC_MMIO_PASS`。
