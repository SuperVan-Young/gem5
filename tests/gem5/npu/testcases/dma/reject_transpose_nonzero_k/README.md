# DMA reject_transpose_nonzero_k

## 测试目的

故意让 transpose 使用带 blocking 的布局，验证 DMA 会拒绝当前模型
尚未支持的 transpose 加 blocked-layout 组合。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：当前 transpose 只接受 `src_k == 0`
  且 `dst_k == 0` 的非 blocked 布局。
- 构造的非法命令改了哪个字段：程序构造一条 transpose 命令，
  把源布局的 `k` 设成 `2`。
- DMA 为什么必须拒绝它：一旦 transpose 接收到 blocked 布局，
  地址重排逻辑就需要同时处理轴交换和块内展开。当前模型没有为这两种
  语义的组合定义实现规则。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`transpose requires src_k == 0 and dst_k == 0`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
