# DMA reject_invalid_blocked_k_h

## 测试目的

故意让 blocked-H 布局的块大小 `k` 不能整除高度 `H`，验证 DMA 会拒绝
无法正确展开的 blocked 布局。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：当源布局按 H 维做 blocking 时，必须满足
  `H % k == 0`。
- 构造的非法命令改了哪个字段：程序构造一条 move-layout 命令，
  让源布局使用 blocked-H，设置 `H=3`、`k=2`。
- DMA 为什么必须拒绝它：如果高度不能被块大小整除，末尾就会出现残缺块，
  DMA 无法用一套固定步长解释整个张量。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`source blocked layout requires H % k == 0`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
