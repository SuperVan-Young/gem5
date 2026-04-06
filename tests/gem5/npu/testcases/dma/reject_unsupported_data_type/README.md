# DMA reject_unsupported_data_type

## 测试目的

故意把 DMA 命令的数据类型编码设成当前模型不支持的值，验证 DMA 会在
解码阶段直接拒绝该命令。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：DMA 只接受当前模型已实现的数据类型编码。
- 构造的非法命令改了哪个字段：程序构造一条 move-layout 命令，
  把 `data_type` 从默认合法值改成 `1`。
- DMA 为什么必须拒绝它：当前模型没有为 `data_type=1` 定义搬运语义。
  继续执行会让解码器在没有明确数据解释规则的前提下处理命令。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`unsupported data_type=1`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
