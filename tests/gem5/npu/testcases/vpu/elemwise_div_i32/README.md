# VPU 整数除法测试

## 测试目的

验证 VPU 的整数除法路径是否稳定，覆盖基础二元除法和目的端写回。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存、一个 `MegaCmdQueue` 和一个 `VPU`。
两个源向量分别放在独立端口，商写回到第三个端口。

## 仿真程序

workload 准备两个 4 元素的整数向量，发射一条 `VDIV` 命令，并等待 golden 结果出现在目的端口。
程序不组合其它算子，也不使用边界混合场景。

## 预期行为

仿真结束时，VPU 的完成计数应为 1，prologue/execute/epilogue/iteration 计数都应为 1，读响应计数应为 2，写响应计数应为 1。
配置脚本应输出稳定的 `VPU_ELEMWISE_DIV_I32_PASS` 标记。
