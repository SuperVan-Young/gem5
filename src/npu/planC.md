# NPU 当前进展与后续收尾规划

本文档承接 [planA.md](/home/wangyukun/workspace/gem5/src/npu/planA.md) 和
[planB.md](/home/wangyukun/workspace/gem5/src/npu/planB.md)，用于总结当前
`src/npu` 代码基线已经完成到什么程度，以及剩余问题应如何分步骤推进。

## 当前项目进展

截至目前，`mega/` 路径已经从“基础 VPU 算子可运行”推进到“线性算子 +
非线性 LUT + 系统联动测试闭环”阶段。

### 已完成项

- `VpuUnit` 已支持：
  - `VADD/VSUB/VMUL/VDIV`
  - `VSCALE/VCVT/VSQRT/VFMA`
  - `VREDUCE_SUM/VREDUCE_MAX`
  - `VLOAD/VSTORE`
  - `VEXP/VSOFTMAX`
- 原先在 `VpuUnit` 内部隐含的 LUT 资源，已经抽成独立
  [LutUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/LutUnit.hh) /
  [LutUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/LutUnit.cc) /
  [LutUnit.py](/home/wangyukun/workspace/gem5/src/npu/mega/LutUnit.py)。
- [SConscript](/home/wangyukun/workspace/gem5/src/npu/mega/SConscript) 已注册
  `LutUnit`，因此 Python 配置可以显式实例化和共享该资源。
- `VSQRT`、`VEXP`、`VSOFTMAX` 已统一变为“向 `LutUnit` 申请资源与时延，
  再执行结果写回”的模式。
- `Softmax` 的功能边界已经拉直为：
  - `reduce_max`
  - `exp(LUT)`
  - `reduce_sum`
  - `normalize`
- 统计与观测接口已经初步分层：
  - `LutUnit` 持有主统计与主 completion tick
  - `VpuUnit` 保留必要镜像接口，兼容现有 config/test 读取方式
- 系统搭建脚本已经支持“多个 VPU 显式共享一个 LUT”。

### 已完成测试闭环

目前已经有对应测试覆盖以下层次：

- 单算子：
  - `vpu_unary`
  - `vpu_softmax`
- 基础系统联动：
  - `system_basic`
  - `system_vpu_dual`
- 近似 SEU 联动路径：
  - `system_pipeline`
  - `system_multiport`

这些测试已经能验证：

- 新 opcode 经 `MegaCmdQueue` 路由正确
- `sync` 顺序不被 LUT 引入后打乱
- `LutUnit` 参与后 completion tick 晚于线性路径
- “两个 VPU + 一个共享 LUT”的基础联动是成立的
- DMA + VPU 线性处理 + Softmax 已具备可执行闭环

## 当前仍未解决的问题

虽然功能和基础 timing 闭环已经建立，但当前实现仍然偏“第一版可运行模型”，
离更稳定、更接近结构化硬件模型还有几步。

### 1. `SEU` 拓扑还没有真正落地

当前 `LutUnit` 已经独立成 SimObject，但还没有收敛成“`SEU` 拥有 2 VPU + 1 DMA
+ 1 LUT”的清晰复合对象图。现在更多是 test/config 手动把这些部件连在同一系统里。

### 2. `LutUnit` 仍是最小资源模型

当前 LUT 资源模型仍然只有：

- 单资源
- 固定 latency
- 单队列
- 无 bank
- 无显式队列深度
- 无复杂仲裁/背压统计

这足以验证功能边界和共享行为，但还不足以研究更细的竞争效应。

### 3. `Softmax` 的阶段边界已明确，但时序还没有分项可视化

逻辑上 `Softmax` 已经拆成 `reduce_max + exp + reduce_sum + normalize`，但现在
对外暴露的时序仍主要是“整条 softmax 命令的 execute/completion”。如果后续要
分析瓶颈，还需要把内部子阶段的 timing 与计数进一步显式化。

### 4. 共享 LUT 的压力测试还不够强

当前测试证明了“共享 LUT 可工作”，但还没有系统覆盖：

- 两个 VPU 长序列交织提交
- 不同长度向量对同一 LUT 排队
- LUT 资源拥塞下 completion 次序与 latency 演化
- 更密集的多端口/多 CPU 同时提交

### 5. 文档与参数语义还需要收敛

目前不少设计意图主要沉淀在代码和测试里，还缺：

- `README` 层面对 `LutUnit` / shared LUT 用法的说明
- 参数含义与推荐默认值说明
- 对“当前 timing 模型属于功能定形版，不代表真实硬件周期模型”的明确标注

## 后续分步骤规划

下面建议按“先拉直结构边界，再增强资源细节，最后补压力和文档”的顺序推进。

## Step 1：整理当前基线并补文档出口，已完成

### 目标

先把目前已经完成的行为写成稳定文档，减少后续继续演进时的认知分叉。

### 建议动作

- 更新 `src/npu/README.md`：
  - 增加 `LutUnit` 的角色说明
  - 说明 `VSOFTMAX` 当前的内部步骤
  - 说明共享 LUT 的连接方式
- 在 `LutUnit.py` / `VpuUnit.py` 参数注释中明确：
  - latency 参数含义
  - 当前模型是单资源单队列
- 把当前已经通过的测试目录列成推荐回归集合

### 退出条件

- 新开发者不翻代码，也能知道当前 non-linear 路径是怎样工作的。

## Step 2：把 `SEU` 组合关系收敛成正式对象图，已完成

### 目标

不再让 test/config 手工拼“共享 LUT”，而是让 `SEU` 成为更清晰的结构边界。

### 建议动作

- 明确 [SpecializedExecutionUnit.py](/home/wangyukun/workspace/gem5/src/npu/mega/SpecializedExecutionUnit.py)
  / `SpecializedExecutionUnit.hh/.cc` 与 `SEU` 外层关系。
- 设计新的组合方式：
  - `SEU`
  - `dma`
  - `vpu0`
  - `vpu1`
  - `lut`
- 第一版不追求复杂自动布线，只要把 ownership 和参数入口收敛即可。

### 退出条件

- 配置脚本可以直接实例化“一个带 2 VPU + 1 DMA + 1 LUT 的 SEU”。

## Step 3：增强 `LutUnit` 的资源模型

### 目标

从“单资源固定延迟”推进到“可研究竞争”的共享部件。

### 建议动作

- 给 `LutUnit` 增加显式队列深度与入队失败/背压统计。
- 支持多 bank 或多 pipeline lane 的最小模型。
- 区分：
  - 请求进入队列
  - 真正开始执行
  - 执行完成
- 增加基本仲裁策略：
  - 先到先服务
  - 可选按 VPU 轮转

### 退出条件

- 两个 VPU 竞争 LUT 时，latency 行为能随参数变化而变化，不再只是简单串行固定偏移。

## Step 4：把 `Softmax` 内部 timing 细节显式化

### 目标

让 `Softmax` 不仅逻辑上复用内部步骤，也能在统计层面被拆开观察。

### 建议动作

- 增加 `Softmax` 子阶段统计：
  - `reduce_max`
  - `exp_requests`
  - `reduce_sum`
  - `normalize`
- 区分：
  - LUT 子阶段 latency
  - 非 LUT 子阶段 latency
- 评估是否需要在 `VpuUnit` 中增加更细粒度的 debug trace tag

### 退出条件

- 可以回答“`Softmax` 变慢是慢在 LUT、规约，还是归一化”。

## Step 5：补更强的系统级压力测试

### 目标

把当前“功能闭环已通”的测试集合升级为“共享资源 regression”集合。

### 建议动作

- 视需要新增 `system_softmax_pipeline/`
- 升级 `system_multiport`，让两个端口都持续提交含 LUT 的命令流
- 新增至少一个“长向量 + 双 VPU 交织提交”的场景
- 在 config 里统一检查：
  - `queueOccupancy()`
  - `isIssueBusy()`
  - `LutUnit` 请求数/命令数
  - completion tick 顺序

### 退出条件

- 共享 LUT 的基本竞争模式已有自动 regression 覆盖。

## Step 6：收敛数值模型与参数标定

### 目标

在结构边界稳定后，再决定是否把数值近似和 timing 参数做得更细。

### 可选方向

- 提高 LUT 表项配置灵活度
- 支持不同插值策略
- 区分 `sqrt` / `exp` / `softmax` 子路径的不同 timing 参数
- 引入更贴近目标硬件的经验参数

### 退出条件

- 模型既保持可解释性，也能服务后续性能研究，而不是只停留在功能模拟。

## 建议执行顺序

建议按下面顺序推进，避免过早掉进细节：

1. 先做 Step 1，补文档和当前基线说明
2. 再做 Step 2，把 `SEU` 组合关系收敛
3. 然后做 Step 3，增强 `LutUnit` 资源模型
4. 接着做 Step 4，把 `Softmax` timing 细化
5. 最后做 Step 5/6，补压力测试和参数标定

## 当前结论

当前项目已经完成了 `planB` 的核心目标：非线性路径不再是隐藏在 `VpuUnit`
内部的宿主机直算，而是具备独立 `LutUnit`、可观测时序、可共享资源、可系统联动
测试的版本。

下一阶段的重点不再是“能不能算出来”，而是：

- 让结构归属更清晰
- 让共享资源模型更像真实硬件
- 让 timing/统计更适合做后续研究和 regression
