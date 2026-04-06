# DMA reject_invalid_blocked_k

## 测试目的

故意让 blocked-W 布局的块大小 `k` 不能整除宽度 `W`，验证 DMA 会拒绝
无法正确展开的 blocked 布局。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：当源布局按 W 维做 blocking 时，必须满足
  `W % k == 0`。
- 构造的非法命令改了哪个字段：程序构造一条 move-layout 命令，
  让源布局采用默认的 blocked-W 形式，并把 `W=4`、`k=3`。
- DMA 为什么必须拒绝它：blocked 布局要求每个块大小一致。
  `W` 不能被 `k` 整除时，最后一个块的大小会不同，DMA 无法得到一致的
  地址展开规则。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`source blocked layout requires W % k == 0`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
