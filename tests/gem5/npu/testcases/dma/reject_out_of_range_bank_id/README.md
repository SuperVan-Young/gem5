# DMA reject_out_of_range_bank_id

## 测试目的

故意给 transpose 命令传入超出 bank 数量上限的源 bank 编号，
验证 DMA 会拒绝不存在的 bank。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：transpose 命令中的 `src_bank_id`
  必须落在当前 DMA 已实现的 bank 编号范围内。
- 构造的非法命令改了哪个字段：程序构造一条 transpose 命令，
  把 `src_bank_id` 设成 `2`。当前 DMA 只有两个 bank，可用编号是
  `0` 和 `1`。
- DMA 为什么必须拒绝它：越界 bank id 无法映射到真实 bank。
  如果接受它，DMA 将无法确定转置过程使用哪块 scratch bank。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`src_bank_id=2 exceeds num_banks=2`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
