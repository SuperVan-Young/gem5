# DMA reject_reserved_cut_dim

## 测试目的

故意把 move-layout 的切分维度编码设成保留值，验证 DMA 会拒绝
未定义的布局枚举。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：move-layout 的 `src_cut_dim` 必须是
  当前已定义的 H、W 或 C 之一，不能使用保留值。
- 构造的非法命令改了哪个字段：程序构造一条合法 move-layout 命令，
  再把 `src_cut_dim` 改成保留值 `3`。
- DMA 为什么必须拒绝它：切分维度决定 blocked 布局如何解释地址。
  一旦枚举值未定义，DMA 就无法知道应该沿哪个维度切块。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`reserved src_cut_dim=3`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
