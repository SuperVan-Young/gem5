# DMA reject_transpose_exceeds_bank_size

## 测试目的

故意让 transpose 需要的 scratch bank 空间超过单 bank 容量，
验证 DMA 会拒绝超容量的转置请求。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：transpose 使用的 bank 暂存空间需求
  不能超过单 bank 容量上限。
- 构造的非法命令改了哪个字段：程序构造一条 transpose 命令，
  把源布局设成 `1 x 1 x 4097`，目标布局设成 `1 x 4097 x 1`，
  从而让所需字节数达到 `4097`。
- DMA 为什么必须拒绝它：当前单 bank 容量只有 `4096` 字节。
  如果仍允许执行，转置临时缓冲将越过 bank 边界。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`transpose required_bytes=4097 exceeds bank_size=4096`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
