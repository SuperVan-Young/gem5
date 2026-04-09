# DMA reject_transpose_equal_dims

## 测试目的

故意让 transpose 交换同一个维度，验证 DMA 会拒绝没有实际交换意义的
转置配置。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：transpose 要求 `transpose_dim_a` 与
  `transpose_dim_b` 指向两个不同的维度。
- 构造的非法命令改了哪个字段：程序构造一条 transpose 命令，
  把 `transpose_dim_a` 和 `transpose_dim_b` 都设成 H。
- DMA 为什么必须拒绝它：交换同一个维度不会形成合法的转置语义。
  允许这种命令只会让“转置”退化成一条字段自相矛盾的 no-op。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`transpose requires transpose_dim_a != transpose_dim_b`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
