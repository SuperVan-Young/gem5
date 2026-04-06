# DMA reject_invalid_destination_address

## 测试目的

故意把 DMA 目的地址设到可访问内存范围之外，验证 DMA 会拒绝
无法映射的写回目标。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：move-layout 命令的 `dst_base_addr`
  必须落在 DMA 可访问的 DRAM 或 SPM 地址区间内。
- 构造的非法命令改了哪个字段：程序构造一条 move-layout 命令，
  把 `dst_base_addr` 设成 `0x10000000`。
- DMA 为什么必须拒绝它：这个地址不属于已映射的 DRAM/SPM 地址空间。
  如果继续执行，DMA 将向未定义目标写入数据，写回位置没有合法归属。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`invalid destination base address`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
