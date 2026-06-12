## 测试目的

对 FlashAttention 的 `QK^T` 典型 matmul 形状做第一步性能校准，固定验证：

- `queryLen = 256`
- `kvLen = 1024`
- `headDim = 128`
- GEMM 形状 `M=256, N=1024, K=128`

同一次 testcase 中同时运行：

- 基线 `mpu_gemm.hh` 单 tile 发射路径
- 新的 `fastMatmul.hh` 命令模板复用路径

并要求两条路径数值一致、自动产出 profiling 结果与性能摘要。

## 仿真系统

使用 gem5 NPU testcase 的单核 RISC-V SE 最小系统，并显式配置：

- `1GHz` 系统时钟
- `256 KiB` SPM
- `25.378787879 GB/s` DRAM 带宽
- `128 x 128` 的 MPU 阵列近似

当前 `MPU v1.1` 的已知约束是：每次 `compute` 结束后，输入 A/B buffer 会自动释放。
因此本 testcase 的 `fastMatmul` 不能做硬件级输入常驻复用，第一版优化重点放在：

- 命令模板复用
- 跨 tile 的 scratch A 复用
- RISC-V 侧低开销发射

## 仿真程序

workload 先构造一组确定性的 `int8` 输入矩阵，然后用 golden GEMM 生成参考输出。

之后依次运行两段 tiled matmul：

1. baseline
2. fast

两段都使用相同的 tile staging 方式：

- 外层按 `row tile`
- 次外层按 `k tile`
- 内层按 `col tile`
- A tile 每个 `(row tile, k tile)` 只搬一次到 scratch
- B tile 每个输出 tile 搬一次到 scratch

也就是说，本 testcase 不把数据重排成本算进 compare，只对比 matmul 发射路径本身。

## 预期行为

仿真通过时应看到：

- `FLASH_ATTENTION_MATMUL_SCENARIO=flash_attention_matmul_q256_kv1024_d128`
- `FLASH_ATTENTION_MATMUL_SHAPE`
  `m=256 n=1024 k=128 tile_m=128 tile_n=128 tile_k=128`
- `FLASH_ATTENTION_MATMUL_BASELINE_END ... status=PASS`
- `FLASH_ATTENTION_MATMUL_FAST_END ... status=PASS`
- `FLASH_ATTENTION_MATMUL_PROFILE_BASELINE ... status=PASS`
- `FLASH_ATTENTION_MATMUL_PROFILE_FAST ... status=PASS`
- `FLASH_ATTENTION_MATMUL_COMPARE ... status=PASS`
- `FLASH_ATTENTION_MATMUL_PASS`
- `MPU_EXIT_CODE=0`

测试后会自动生成：

- 原始 profile log
- 解析后的 profile JSON
- profile HTML
- testcase-local `performance_summary.json`

用于后续继续做 matmul 与 FlashAttention 的性能收敛。
