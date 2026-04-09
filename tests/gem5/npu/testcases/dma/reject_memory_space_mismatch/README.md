# DMA reject_memory_space_mismatch

## 测试目的

故意让命令里声明的内存空间与真实基地址所在空间不一致，验证 DMA 会拒绝
这种自相矛盾的命令。

## 仿真系统

该 testcase 使用单 CPU 的 timing 系统，系统中挂接 MegaCmdQueue、
DMA、一段可访问的 DRAM 地址空间和一块 SPM。程序只发射一条 DMA
宏命令，不混入其他 reject 场景。

## 仿真程序

- 被故意违反的 DMA 约束：`mode_cfg` 中编码的源内存空间必须与
  `src_base_addr` 实际所属地址空间一致。
- 构造的非法命令改了哪个字段：程序把 `src_base_addr` 设成
  `0x60001000`，这个地址落在 SPM；但又把 `mode_cfg` 中的源空间
  编码成 DRAM。
- DMA 为什么必须拒绝它：同一条命令不能同时宣称“源在 SPM”又
  “源在 DRAM”。如果继续执行，DMA 将按错误的内存域解释地址。

## 预期行为

gem5 应触发 panic，stderr 中应出现关键错误信息
`invalid source base address 0x60001000`。
如果命令意外没有触发 panic，`config.py` 会主动报错，避免把
“静默通过”误判成测试通过。
