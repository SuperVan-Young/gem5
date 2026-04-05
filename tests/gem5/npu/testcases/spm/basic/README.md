# SPM 基础访问测试

## 测试目的

验证 `ScratchpadMemory` 的基础可读写行为是否稳定，包括字节访问、字访问、覆盖写和简单非对齐访问。

## 仿真系统

该 testcase 使用单 CPU timing 系统、默认物理内存和一个 `SPM`。
不连接 `MegaCmdQueue`、`SEU`、`DMA` 或其他 NPU 执行单元。

## 仿真程序

workload 直接对 `SPM` 地址空间进行字节和字读写，随后覆盖写同一片区域，并验证简单的非对齐字节访问。

## 预期行为

仿真应以正常进程退出结束，退出原因应正确、退出码应为 0，并输出 `SPM_BASIC_PASS`。
