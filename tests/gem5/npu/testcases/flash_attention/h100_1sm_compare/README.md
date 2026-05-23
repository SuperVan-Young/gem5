## 测试目的

提供 FlashAttention H100 单 SM compare/calibration testcase 的第一版完整链路：

- workload 真正执行 `QK -> online softmax -> PV`
- `QK` 与 `PV` 的 tensor path 走当前 `MpuUnit`
- `online softmax` 继续显式走 CPU helper
- testcase 结束后自动生成 profile artifact 和 compare-oriented JSON 摘要

当前目标是打通“可测量、可校验、可继续校准”的 compare 链路，不把结果误写成
“已经精确复现 H100 1SM 端到端 latency”。

## 仿真系统

该 testcase 继续使用 gem5 NPU testcase 的 RISC-V SE 模式最小系统，但
不再直接 `runpy` 复用 `mpu_proto.py`，而是在 testcase-local `config.py`
里显式组装一份 H100 1-SM 近似参数，避免改动公共 infra。

当前近似映射如下：

- 片上 buffer 容量：`ScratchpadMemory` 配成 `256 KiB`
  这是本 testcase 对“单 SM 可用片上 buffer 约 256KB”的主映射。
- DRAM 带宽：`lowmem` 配成 `25.378787879 GB/s`
  该值直接取自 `3.35 TB/s / 132`，按十进制带宽写入配置。
- DRAM 映射窗口：进程侧把 `0x20000000` 起始 DRAM 映射放大到 `32 MiB`
  这不是 H100 参数本身，而是给后续真正跑 compare/calibration workload
  预留足够地址空间，避免继续沿用 `64 KiB` 的 demo 级窗口。

当前配置里，近似 `VECTOR` 和近似 `TENSOR` 的参数分工是：

- `VECTOR`：
  当前没有单独的 H100 vector pipe / vector core 模型，也没有单独的
  “vector TFLOPS” 参数。
- `VECTOR`：
  现阶段只能把 `RiscvTimingSimpleCPU`、系统时钟、`256 KiB` SPM、
  SPM/DRAM 时延和 `25.378787879 GB/s` DRAM 带宽，当作量化、dequant、
  online softmax 等非-MPU 阶段的粗粒度共享资源近似。
- `TENSOR`：
  `MpuUnit(array_dim=128)` 近似“一次能吃下 128x128x128 tile 的方阵 MAC
  阵列”，A/B/C buffer 分别设成 `16 KiB / 16 KiB / 64 KiB`，对应单个
  `128 x 128` int8 A tile、单个 `128 x 128` int8 B tile 和单个
  `128 x 128` int32 C tile 的容量上限。
- `TENSOR`：
  `num_mem_side_ports=2` 以及 MPU 自带的 command / load / drain latency
  仍沿用现有模型，表示 tensor path 的访存并发和流水线行为，但这仍只是
  gem5 现有 `MpuUnit` 的近似，不是 H100 tensor core 微结构复刻。

当前模型限制必须明确：

- 现有配置不能直接表达 `67 TFLOPS / 132` 这一类 H100 单 SM 的
  `VECTOR` 峰值指标。当前没有独立 vector 执行单元参数，也没有直接以
  TFLOPS 标定的软件/向量阶段模型。
- 现有配置也不能直接表达 `2000 TFLOPS / 132` 这一类 H100 单 SM 的
  `TENSOR` 峰值指标。`MpuUnit` 只能通过 `array_dim`、tile 尺寸和内部固定
  的 `compute latency = k + m cycles` 规则间接产生产能，因此不是精确的
  FLOPS 标定模型。
- 因为 `MpuUnit` 当前是方阵 `array_dim` 抽象，它更像“单 tile 方阵 MAC”
  的近似，而不是 H100 tensor core 的真实 MMA instruction shape。
- `256 KiB` 这里只是把 testcase 可见片上 buffer 收敛到 H100 量级；
  它不代表已经精确还原了寄存器文件、shared memory、tensor core staging
  buffer 之间的真实容量切分。

## 仿真程序

workload 使用 `flash_attention.hh` helper 路径，固定 compare 场景名
`flash_attention_h100_1sm_compare`，并保持：

- 默认 shape：`seq_len=1024`、`head_dim=128`
- compare 契约：`batch=1`、`heads=1`、`sms=1`
- 映射说明：`one_sm_per_head_per_batch`

当前实现同时保留一个 `2x3x4` single-tile probe：

- probe 只作为最小硬件活性检查
- compare 结果不再复用 probe 的统计
- probe 与 compare 在 profile 中通过独立 `sync_indicator` 做区分

完整 compare 路径的实现口径是：

- `QK` 采用 `128x128x128` tile，在 SPM scratch 上逐 tile 发射
- `PV` 采用 `128x128x128` tile，并沿 `k` 维做分块累加
- 全量 `Q/K/V`、中间 `scores/p/out` 和 metadata 常驻 workload DRAM 缓冲区
- `SPM=256 KiB` 只承载当前 tile scratch，不承载完整 tensor

## 预期行为

仿真通过时应至少看到以下稳定输出：

- `FLASH_ATTENTION_H100_1SM_COMPARE_SCENARIO=...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_SHAPE ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_CONTRACT ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_PROBE ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_BEGIN ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_FLOPS ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_COMPARE_END ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_PERF_SUMMARY ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_SYSTEM ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_TENSOR_APPROX ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_VECTOR_APPROX ...`
- `FLASH_ATTENTION_H100_1SM_COMPARE_PASS`
- `MPU_EXIT_CODE=0`

`FLASH_ATTENTION_H100_1SM_COMPARE_PERF_SUMMARY` 当前的 compare 口径是：

- `compare_cycles`：`QK` stage span + `PV` stage span
- `compare_latency`：仅统计 MPU `COMPUTE` macro duration
- `compare_busy`：MPU busy cycles
- `compare_idle`：仅统计 `QK/PV` stage 内部 gap，不混入 CPU softmax 空档
- `compare_spm_wait`：MPU 累计 SPM wait cycles
- `softmax_scope=cpu_helper`
- `excluded_from_compare_latency=true`

本 testcase 现在还会在测试后自动执行目录内的 `summarize_perf.py`，把本次
运行的关键结果收口到 testcase-local `profile/` 目录，避免人工去找
`tests/testing-results/...` 的临时 run 目录。稳定产物如下：

- `profile/h100_1sm_compare.npu_profile.log`
  gem5 `--debug-file` 直接落盘的原始 `NPUProfile` log。
- `profile/h100_1sm_compare.npu_profile.json`
  由现有 profile artifact verifier 生成的机器可读 profiling 结果。
- `profile/h100_1sm_compare.npu_profile.html`
  由现有 profile artifact verifier 生成的 HTML profiling 报告。
- `profile/h100_1sm_compare.simout.txt`
  verifier 在测试结束后从本次 `simout.txt` 复制出来的稳定 stdout artifact。
- `profile/h100_1sm_compare.performance_summary.json`
  本次新增的性能摘要文件，至少包含：
  `scenario`、`seq_len`、`head_dim`、`qk_flops`、`pv_flops`、
  `total_flops`、`compare_cycles`、`compare_latency`、`compare_macs`、
  `compare_busy`、`compare_idle`、`compare_spm_wait`、
  `compare_scope`、`softmax_scope`，以及保留为原始来源字段的
  `mpu_summary` 和 `profiling`。

查看方式：

- 看 profiling 原始日志与摘要 JSON，直接打开本目录 `profile/` 下对应文件即可。
- 看可视化 profiling，打开 `profile/h100_1sm_compare.npu_profile.html`。
- 看本次收口后的机器可读性能结果，打开
  `profile/h100_1sm_compare.performance_summary.json`。

当前 testcase 仍然只是 `MpuUnit` tile-scale approximation：

- 不是 H100 Tensor Core 指令级复刻
- 不是已校准到 `67/132 TFLOPS` 或 `2000/132 TFLOPS` 的精确模型
- compare 数据当前只适合作为后续校准闭环的输入，不应直接等同于真实 H100
  1SM 端到端 latency
