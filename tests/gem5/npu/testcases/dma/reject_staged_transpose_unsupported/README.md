# DMA reject_staged_transpose_unsupported

## 测试目的
验证 DMA 会显式拒绝非 legacy 的 staged transpose 命令，而不是落到未定义的共享执行路径。
