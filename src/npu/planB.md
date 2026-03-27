# VPU 非线性 LUT 路径分阶段实施规划

本文档承接 [planA.md](/home/wangyukun/workspace/gem5/src/npu/planA.md)，聚焦把
`sqrt/exp/softmax` 从“宿主机函数式直算”收敛到“显式 LUT 资源 + 可观测时序”的实现。

当前代码基线已经具备：

- `VADD/VSUB/VMUL/VDIV`
- `VSCALE/VCVT/VSQRT/VFMA`
- `VREDUCE_SUM/VREDUCE_MAX`
- `VLOAD/VSTORE`

还缺的关键点是：

- `sqrt/exp` 没有独立 LUT 资源和排队时序。
- `Softmax` 尚未复用统一非线性路径。
- 测试里还没有把“功能正确性”和“完成时间特征”拆开验证。

## 目标

第一阶段目标不是一步做成高精度数学库，而是先把“结构和时序边界”落准：

1. 在 `VpuUnit` 内部引入显式 LUT 资源模型。
2. 新增 `VEXP` 与 `VSOFTMAX` opcode。
3. `VSQRT/VEXP/VSOFTMAX` 统一走 LUT 资源，而不是直接 `std::sqrt/std::exp`。
4. `Softmax` 复用同一套 `exp` LUT helper，而不是单独硬编码另一条路径。
5. 暴露 LUT 请求数、命令数、线性/非线性执行 latency 与 completion tick，
   让测试能检查时序差异。

## 分阶段实施

## 阶段 0：抽出最小可用 LUT 资源

### 目标

先把“非线性算子要占用一个资源、并带来额外时延”建出来。

### 建议实现

- 在 [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh) /
  [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc) 中增
  加 `LutResource`：
  - `availableTick`
  - `requestCount`
  - `commandCount`
- 增加以下参数：
  - `lut_range_reduction_latency`
  - `lut_lookup_latency`
  - `lut_interpolation_latency`
  - `lut_normalize_latency`
  - `lut_table_entries`
- 第一版不强行拆成单独 SimObject，先做成 `VpuUnit` 内部资源即可。
  这样改动面小，但接口上保留后续独立 SimObject 的演进空间。

### 退出条件

- 非线性命令的 `execute()` latency 明显大于普通逐元素命令。
- Python config 能直接读出 LUT 请求数和非线性 latency。

## 阶段 1：把 `sqrt` 改为 LUT 路径

### 目标

让 `VSQRT` 不再通过执行时 `std::sqrt` 直接出结果。

### 建议实现

- 使用范围规约把输入规约到固定区间，例如 `[1, 4)`。
- 以 `lut_table_entries` 生成离线表。
- 第一版使用“离散表 + 线性插值”。
- 负数、`NaN`、`Inf` 的处理规则在代码与测试中固定。

### 测试

- 扩展 [vpu_unary](/home/wangyukun/workspace/gem5/tests/gem5/npu/vpu_unary)：
  - 正确性：`sqrt(1/4/9/16)`
  - 时序：`lastLutExecuteLatency > lastLinearExecuteLatency`

### 退出条件

- `VSQRT` 功能测试稳定通过。
- LUT 计数器能看到该命令确实占用了 LUT 资源。

## 阶段 2：新增 `VEXP`

### 目标

建立 `Softmax` 所需的基础一元非线性算子。

### 建议实现

- 新增 `VEXP` opcode 与软件可见 helper。
- 使用 `x = n * ln2 + r` 的范围规约，把表区间固定到 `[0, ln2]`。
- 第一版仍采用“查表 + 线性插值 + 固定额外时延”。

### 测试

- 继续扩展 [vpu_unary](/home/wangyukun/workspace/gem5/tests/gem5/npu/vpu_unary)：
  - 正确性：`exp(-1/-0.5/0/1)`
  - 时序：LUT 请求总数应等于 `sqrt + exp` 参与元素数之和

### 退出条件

- `VEXP` 独立可测。
- `exp` 的数值误差阈值已经被测试固定。

## 阶段 3：用 `VEXP` 路径完成 `Softmax`

### 目标

实现 `VSOFTMAX`，并明确它不是另一份单独的非线性实现，而是复用
`reduce_max + exp_lut + sum + normalize`。

### 建议实现

- `Softmax` 固定 `Float32`。
- 执行步骤：
  1. `reduce_max`
  2. 对每个元素执行 `lutExp(x - max)`
  3. `reduce_sum`
  4. 归一化写回
- 其中 LUT 资源只服务 `exp` 子步骤，但 `VSOFTMAX` 的对外执行 latency
  要显式包含这段 LUT 占用。

### 测试

- 新增 [vpu_softmax](/home/wangyukun/workspace/gem5/tests/gem5/npu/vpu_softmax)：
  - `ramp` 输入
  - `flat` 输入
  - workload 内按误差阈值判定 PASS/FAIL
  - config 校验：
    - `lutRequestCount`
    - `lutCommandCount`
    - `lastSoftmaxExecuteLatency`
    - `lastSoftmaxCompletionTick`

### 退出条件

- `Softmax` 可独立通过功能测试。
- `Softmax` 的 LUT 资源消耗可观测，且和元素数一致。

## 阶段 4：资源模型细化

### 目标

把第一版“固定延迟串行资源”演进成更接近硬件的共享部件。

### 后续方向

- 从 `VpuUnit` 内部资源迁移到独立 `SimObject` 或 `SEU` 共享资源。
- 增加：
  - 带宽
  - 可并发请求数
  - 队列深度
  - 背压/冲突统计
- 允许多个 `VpuUnit` 或未来 `SEU` 子部件共享同一 LUT。

### 退出条件

- 非线性算子时序开始受共享争用影响。
- 系统级测试可以观察到多设备竞争 LUT 的行为差异。

## 测试要求

每一阶段都至少要有两类检查：

- 功能正确性
- 时序特征

时序特征的最低要求：

- 普通逐元素算子与 LUT 非线性算子的完成 tick 或 execute latency 可区分。

## 当前建议落地顺序

1. 先在 `VpuUnit` 内建最小 LUT 资源。
2. 先改 `VSQRT`，再加 `VEXP`。
3. 最后让 `VSOFTMAX` 复用 `VEXP` 路径。
4. 资源抽象独立化放在第二批，不阻塞当前功能闭环。
