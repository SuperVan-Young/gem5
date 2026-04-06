# DMA reject_reserved_transpose_dim

## 测试目的

故意把 transpose 的维度字段设成保留值，验证 DMA 会拒绝
未定义的转置维度枚举。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：transpose 的 `transpose_dim_a` 和
  `transpose_dim_b` 必须是当前已定义的张量维度，而不能是保留值。
- 构造的非法命令改了哪个字段：程序构造一条 transpose 命令，
  把 `transpose_dim_a` 改成保留值 `3`。
- DMA 为什么必须拒绝它：转置维度决定 DMA 如何重排索引。
  当维度编码未定义时，DMA 根本无法知道要交换哪两个轴。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`reserved transpose_dim_a=3`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
