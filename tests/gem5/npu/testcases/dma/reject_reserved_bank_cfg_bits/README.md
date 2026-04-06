# DMA reject_reserved_bank_cfg_bits

## 测试目的

故意在 transpose 的 bank 配置字里写入保留位，验证 DMA 会拒绝
带有未定义附加编码的 transpose 命令。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：transpose 命令的 `bank_cfg` 只能包含
  已定义的源 bank 和目的 bank 字段，保留位必须为 `0`。
- 构造的非法命令改了哪个字段：程序先构造一条合法 transpose 命令，
  再把 `bank_cfg` 额外或上 `0x100`，故意点亮保留位。
- DMA 为什么必须拒绝它：保留位被置位后，命令就不再对应唯一明确的
  bank 配置语义。允许执行等于让 DMA 接受未定义编码。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`reserved bank_cfg bits set for transpose`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
