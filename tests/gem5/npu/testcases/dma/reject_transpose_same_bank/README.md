# DMA reject_transpose_same_bank

## 测试目的

故意让 transpose 的输入 bank 和输出 bank 设成同一个 bank，验证 DMA
会拒绝可能导致自覆盖的 bank 配置。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：transpose 要求 `src_bank_id` 与
  `dst_bank_id` 不同，避免转置过程读写同一 scratch bank。
- 构造的非法命令改了哪个字段：程序构造一条 transpose 命令，
  把 `src_bank_id` 和 `dst_bank_id` 都设成 `1`。
- DMA 为什么必须拒绝它：转置需要一边读取源 bank，一边写入目的 bank。
  如果两者是同一个 bank，转置过程会在读完前覆盖掉还没消费的数据。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`transpose requires src_bank_id != dst_bank_id`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
