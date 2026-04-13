# DMA reject_staged_fill_unsupported

## 测试目的
验证 DMA 会显式拒绝非 legacy 的 staged fill 命令，而不是把它静默退化成 legacy 路径。
