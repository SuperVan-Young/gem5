# NPU 联动测试补齐规划

本文档承接 [plan.md](/home/wangyukun/workspace/gem5/tests/gem5/npu/plan.md)，
专门规划“随着 `planA.md` 中 VPU 新功能落地，系统级联动测试要怎样补齐”。

当前现状：

- 单算子目录已经覆盖了 `elemwise / unary / fma / reduce / loadstore`。
- 系统级目录仍然以旧 `EXEC` 语义或基础数据流为主。
- `Softmax`、LUT 非线性算子、`VLOAD/VSTORE` 新语义，还没有形成系统级场景。

因此第二阶段的重点不是再堆更多单算子 case，而是把“命令队列 + DMA + 多个
VPU + sync + SPM/DRAM 数据流”的联动场景补齐。

## 总目标

新增一批面向系统级的 NPU 测试，使下列能力都能被自动验证：

1. `MegaCmdQueue` 正确路由新 opcode。
2. `DMA`、`VLOAD/VSTORE` 与逐元素/规约/Softmax 能按 sync 顺序联动。
3. 两个 `VpuUnit` 共享 SPM 时，结果与软件参考一致。
4. LUT 非线性路径在系统级场景下可观测，至少能看见计数器和 completion tick。

## 规划原则

- 优先新增系统场景，不重复单算子目录已有覆盖。
- 每个 workload 必须自己算软件参考结果。
- 每个 config 除 PASS marker 外，还要检查：
  - `queueOccupancy()`
  - `isIssueBusy()`
  - 每个参与设备的 `completedCmdCount()`
- 对含 `Softmax` 的场景，workload 内做浮点容差判断，config 不做 bit-exact。

## 阶段 1：升级 `system_basic`

### 目标

让最基础的一体化测试先使用真正的 VPU opcode，而不是只依赖 legacy `EXEC`。

### 建议新增场景

- 一个 `VADD` 或 `VMUL`
- 一个 `VSQRT` 或 `VEXP`
- 一个 sync wait 把两条命令串起来

### 验证点

- 新 opcode 能经 `MegaCmdQueue` 正常转发。
- 非线性命令的计数器生效。
- 队列 drain 和 sync_done 仍然稳定。

## 阶段 2：升级 `system_vpu_dual`

### 目标

让双 VPU 场景不再只验证“两个实例都能收命令”，而是验证不同算子组合。

### 建议场景

- `vpu0`: `VLOAD -> VADD -> VSTORE`
- `vpu1`: `VLOAD -> VSQRT/VEXP -> VSTORE`
- 两边共享同一条控制队列，通过不同 sync indicator 交错调度

### 验证点

- `device_id` 路由稳定。
- 两个 VPU 的计数器各自匹配。
- 非线性 VPU 的 completion tick 晚于线性 VPU。

## 阶段 3：升级 `system_pipeline`

### 目标

把当前 DMA 种子数据流测试升级成“DMA + 线性算子 + Softmax”的完整流水。

### 建议场景

1. DMA 从 DRAM 搬输入到 SPM
2. `vpu0` 做 `reduce_max` 或简单预处理
3. `vpu1` 做 `VSOFTMAX`
4. DMA 把结果搬回 DRAM

### 参考目录

- [system_pipeline](/home/wangyukun/workspace/gem5/tests/gem5/npu/system_pipeline)
- [vpu_softmax](/home/wangyukun/workspace/gem5/tests/gem5/npu/vpu_softmax)

### 验证点

- DRAM 最终内容与软件参考一致。
- `vpu1.lutRequestCount()` 与元素数匹配。
- DMA/VPU/Queue 都在结束时 drain。

## 阶段 4：补一个 `system_softmax_pipeline/`

### 目标

如果不想把现有 `system_pipeline` 改得过重，单独做一个更聚焦的系统级
`Softmax` 场景。

### 推荐目录

- `tests/gem5/npu/system_softmax_pipeline/`

### 建议拓扑

- 1 CPU
- 1 MegaCmdQueue
- 1 DMA
- 2 VPU
- 1 SPM

### 建议命令流

1. DMA: DRAM -> SPM
2. sync wait
3. VPU0: 可选预处理 `VSCALE` / `VADD`
4. VPU1: `VSOFTMAX`
5. DMA: SPM -> DRAM
6. sync_done

### 验证点

- 数值结果正确
- 非线性路径可观测
- sync 顺序正确

## 阶段 5：补 `system_multiport` 的新功能覆盖

### 目标

把多 CPU/多端口控制压力与新 VPU 功能结合起来。

### 建议场景

- CPU0 提交 DMA 与线性算子命令
- CPU1 提交 `VSQRT/VEXP/VSOFTMAX`
- 两个 CPU 共享 `MegaCmdQueue`

### 验证点

- 不同 CPU 端口提交的新 opcode 都能正确进入共享队列
- sync wait 不会被跨端口提交打乱
- 最终 SPM/DRAM 状态仍然与软件参考一致

## 推荐新增目录

- `system_vpu_dual_ops/`
- `system_softmax_pipeline/`
- `system_multiport_softmax/`

如果希望控制工作量，最少应先完成：

1. 升级 `system_basic`
2. 升级 `system_pipeline` 或新增 `system_softmax_pipeline`
3. 升级 `system_multiport`

## 每个系统测试都应新增的观测量

对于参与的新 VPU 测试配置，建议统一打印：

- `lutRequestCount()`
- `lutCommandCount()`
- `lastLinearExecuteLatency()`
- `lastLutExecuteLatency()`
- `lastSoftmaxExecuteLatency()`
- `lastLinearCompletionTick()`
- `lastLutCompletionTick()`

这样后续系统级 timing regression 不必重新翻 workload 日志。
