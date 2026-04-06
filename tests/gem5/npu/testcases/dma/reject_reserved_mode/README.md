# DMA reject_reserved_mode

## 测试目的

故意把 DMA 命令的 mode 字段设成保留值，验证解码阶段会直接拒绝
不属于任何已实现操作类型的命令。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：DMA `mode` 只能编码成当前已实现的
  move-layout、transpose 或 fill。
- 构造的非法命令改了哪个字段：程序构造一条 DMA 命令，把 `mode`
  直接写成保留值 `3`。
- DMA 为什么必须拒绝它：mode 决定整条命令该按哪套字段语义解码。
  如果 mode 本身未定义，后续所有字段解释都会失去依据。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`unsupported mode=3`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
