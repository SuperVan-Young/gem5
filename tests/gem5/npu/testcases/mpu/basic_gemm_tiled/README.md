# basic_gemm_tiled

## 测试目标
验证基础 MPU GEMM 工具在 `tile_k == K` 约束下，能够按 `M/N`
方向分块执行多 tile 的 INT8xINT8->INT32 矩阵乘法。

## 覆盖范围
- 使用当前 MPU 命令流：`mvin` / `load` / `compute` / `drain` / `mvout`
- 行主序二维矩阵
- 不依赖 overlap 作为正确性前提

## 通过条件
- 最终 C 矩阵与 golden GEMM 结果一致
- 统计摘要显示 4 个 tile，共 28 条命令
