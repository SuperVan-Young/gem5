# VPU 功能实现规划

本文档面向 `src/npu/mega/VpuUnit.*` 的后续扩展，目标是在当前已经支持的：

- 按 `readMask` 从 SPM 读
- 按 `writeMask` 写回 SPM
- `repetition` 重复执行
- 完成后同步信号

基础上，逐步实现真正需要的 VPU 功能：

- `VADD`
- `VSUB`
- `VMUL`
- `VDIV`
- `VFMA`
- `VLOAD`
- `VSTORE`
- `reduce`
- `sqrt`
- `scale`
- `Softmax`
- `int/float` 转换
- 更一般的浮点运算

## 1. 现状与约束

当前代码的真实边界如下：

- 核心实现集中在 [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc) 和 [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)。
- `VpuUnit` 只接受 `device_type = 0x2`、`op_code = 0x0`。
- 现有“执行”本质上是对选中 SPM slot 做 32-bit 读，构造一个 synthetic signature，再把结果写回。
- 命令通用格式由 [common.hh](/home/wangyukun/workspace/gem5/tests/gem5/npu/utils/cmd/common.hh) 提供；VPU 专用字段目前还没有独立 helper。
- 基础执行框架来自 [SpecializedExecutionUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/SpecializedExecutionUnit.hh) 和 [SpecializedExecutionUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/SpecializedExecutionUnit.cc)。
- 现有 VPU 测试 [vpu_test_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/vpu/src/vpu_test_riscv.c) 验证的是“mask/repetition/sync/双实例”，不是算子语义。

因此，不建议直接把所有算子硬塞进当前 `op_code = 0` 的语义里。第一步应先补齐“命令编码 + 操作数描述 + 测试骨架”。

## 2. 按实现难度排序

建议按下面顺序实现：

1. 指令编码与 VPU 数据通路重构
2. `VADD` / `VSUB` / `VMUL`
3. `scale` / `int-float conversion` / `sqrt`
4. `VDIV`
5. `VFMA`
6. `reduce`
7. `VLOAD` / `VSTORE`
8. `Softmax`
9. 更一般的浮点运算框架

排序依据：

- 先做逐元素算子，最容易复用当前“读操作数 -> execute -> 写回”框架。
- `VDIV`、`sqrt`、类型转换涉及异常值、舍入和边界行为，比加减乘更难。
- `VFMA` 依赖三输入和更完整的浮点路径，适合放在二元算子后。
- `reduce` 需要跨 lane 聚合，结果形态和当前逐 port 写回模型不同。
- `VLOAD/VSTORE` 如果定义为显式访存指令，会引入新的地址生成与存储布局问题。
- `Softmax` 是组合算子，天然依赖 `reduce/max`、`exp`、`sum`、`scale/div`。
- “更一般的浮点运算”应放在最后，避免一开始把指令集和执行器做得过于抽象。

## 3. 总体设计原则

### 3.1 先明确 VPU 指令格式

建议保留现有公共头 word0-word4 的含义，并从 `word5` 开始定义 VPU 专用字段：

- `word1`: `readMask`
- `word2`: `writeMask`
- `word3`: `repetition`
- `word4`: 先从“保留字”改为 `flags`
- `word5`: `elem_count`
- `word6`: `src0_offset_bytes`
- `word7`: `src1_offset_bytes`
- `word8`: `src2_offset_bytes` / `scalar_bits`
- `word9`: `dst_offset_bytes`
- `word10`: `src_stride_bytes`
- `word11`: `dst_stride_bytes`
- `word12`: `data_type`
- `word13`: `reduce/rounding/subop`

这样可以兼容两类模式：

- 逐元素算子：从 SPM 某些区域读取向量，写回另一区域。
- 规约/复合算子：读取一个向量，写回标量或较小向量。

### 3.2 明确 `VLOAD/VSTORE` 的边界

建议第一版把 `VLOAD/VSTORE` 定义为：

- `VLOAD`: 从 SPM 地址区间读入 VPU 临时操作数缓冲区
- `VSTORE`: 把 VPU 临时结果缓冲区写回 SPM

不建议第一版让 VPU 直接承担 DRAM <-> SPM 的搬运语义，否则会和 `DmaUnit` 职责重叠。

### 3.3 浮点行为先固定为 FP32

第一阶段浮点算子统一按 FP32 做：

- 输入输出先支持 `int32` 与 `fp32`
- 舍入先使用宿主机 `float`
- `NaN` / `Inf` / 除零 / 负数开方 的语义先在文档和测试中写死

等 FP32 语义稳定后，再扩到“更一般的浮点运算”。

## 4. 分阶段实施

## 阶段 0：指令编码、解码与测试骨架

### 目标

不先实现新算子，先把 VPU 从“只有一个 synthetic op”重构成“可扩展 opcode + 可扩展 operand descriptor”的框架。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 增加 `enum class Opcode`
  - 增加 `enum class DataType`
  - 增加 VPU 专用解码结构，如 `DecodedVpuOp`
  - 增加读取/写入 `int32`、`fp32`、向量片段的 helper
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - 将 `validateCommand()` 从“只允许 op0”改为“按 opcode 分派”
  - 新增 `decodeVpuOp()`、`loadVectorOperands()`、`storeVectorResult()` 等 helper
  - 保留旧 `EXEC` 语义作为回归基线，避免一次性推翻现有测试
- [SpecializedExecutionUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/SpecializedExecutionUnit.hh)
  - `ActiveExecution` 增加 VPU 子类可复用的临时 buffer 元数据入口
- [SpecializedExecutionUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/SpecializedExecutionUnit.cc)
  - 如有必要，放宽单次 request size 与按 port 缓冲组织方式，支持一条 VPU 命令读取一段连续数据，而不是只读单个 32-bit slot
- [common.hh](/home/wangyukun/workspace/gem5/tests/gem5/npu/utils/cmd/common.hh)
  - 保留公共字段
  - 不把 VPU 私有字段继续堆在裸 `setWord()` 调用里
- 新增 [vpu.hh](/home/wangyukun/workspace/gem5/tests/gem5/npu/utils/cmd/vpu.hh)
  - 定义 VPU opcode、datatype、flags
  - 提供 `build_vpu_elemwise_cmd()`、`build_vpu_reduce_cmd()`、`build_vpu_load_store_cmd()` helper

### 测试改动

基于现有 [tests/gem5/npu/vpu](/home/wangyukun/workspace/gem5/tests/gem5/npu/vpu) 做两件事：

- 保留现有 `vpu_basic`，继续作为旧语义回归测试。
- 新增 `vpu_decode_smoke`：
  - 目录建议：`tests/gem5/npu/vpu_decode_smoke/`
  - workload 只发几条不同 opcode/datatype 的命令
  - 暂时不验证算术正确性，只验证：
    - 命令可被正确接受
    - 非法字段会 panic 或拒绝
    - 旧 opcode 仍能通过

### 退出条件

- 新旧命令格式都能稳定运行。
- 之后实现新算子时，不需要再改命令格式。

## 阶段 1：实现 `VADD / VSUB / VMUL`

### 目标

先打通最基本的二输入逐元素算子。这一阶段是后续大多数功能的公共底座。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 定义 `VADD/VSUB/VMUL` opcode
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - 在 `execute()` 中按 opcode 分派到逐元素实现
  - 支持从 SPM 读取 `src0/src1`
  - 支持按 `elem_count + stride + data_type` 解释输入
  - 写回 `dst`
  - 对 `int32` 与 `fp32` 分别实现软件语义

### 测试改动

推荐新建 `tests/gem5/npu/vpu_elemwise/`，复用 `vpu/` 的测试模式：

- `src/vpu_elemwise_riscv.c`
  - seed 多段 SPM 区域作为 `src0/src1/dst`
  - 软件计算参考结果
  - 依次发 `VADD/VSUB/VMUL`
  - 对 `int32`、`fp32` 各测一组
- `configs/vpu_elemwise.py`
  - 复用 `NPUTestSystemBuilder`
  - 至少检查：`completedCmdCount`、`executeCount`、读写响应数、队列 drain
- `test_vpu_elemwise.py`
  - 只匹配稳定 PASS marker

建议覆盖的场景：

- `elem_count = 1/4/16`
- 非重叠源/目的地址
- `repetition > 1`
- `readMask/writeMask` 只启用部分 port
- `int32` 与 `fp32` 各一套基准输入

### 退出条件

- `VADD/VSUB/VMUL` 具备稳定语义。
- 后续 unary、fma、reduce 可复用相同 operand/result 组织。

## 阶段 2：实现 `scale`、`int/float conversion`、`sqrt`

### 目标

补齐一元算子和标量混合算子的基础能力。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 增加 `VSCALE`、`VCVT_I2F`、`VCVT_F2I`、`VSQRT`
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - 增加一元执行路径
  - `scale` 复用 `src0 + scalar_bits`
  - conversion 明确截断/舍入规则
  - `sqrt` 明确对负数输入的行为

### 测试改动

新增 `tests/gem5/npu/vpu_unary/`：

- 在 `src/vpu_unary_riscv.c` 中加入：
  - `scale(fp32)`
  - `int32 -> fp32`
  - `fp32 -> int32`
  - `sqrt(fp32)`
- 参考结果直接在 workload 中按同一规则计算。

建议覆盖的边界值：

- `0`
- 负数
- 大整数
- 小数
- `NaN/Inf` 是否支持，如果支持，测试必须显式写出期望

### 退出条件

- 一元算子的数据类型流转路径稳定。

## 阶段 3：实现 `VDIV`

### 目标

单独处理除法，因为它的边界行为比加减乘复杂。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 增加 `VDIV`
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - `int32` 除法规则
  - `fp32` 除法规则
  - 对除零、溢出、`NaN/Inf` 的行为做统一处理

### 测试改动

扩展 `tests/gem5/npu/vpu_elemwise/`，增加 `div` case，避免新建过多目录。

必须覆盖：

- 正常整除
- 非整除整数
- 浮点小数除法
- 除零
- 被除数为零
- 正负数组合

### 退出条件

- `VDIV` 的特殊值行为在测试里完全固定下来。

## 阶段 4：实现 `VFMA`

### 目标

在已经具备二元算子和标量/浮点基础后，加入三输入算子。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 增加 `VFMA`
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - 支持三输入读取：`src0 * src1 + src2`
  - 明确是否只支持 `fp32`
  - 明确是否采用“真正 fused”还是“先乘后加”的简化语义

建议第一版：

- 只支持 `fp32`
- 语义明确写成“使用宿主机单精度执行一次 fused/或非 fused，二选一并固定”

### 测试改动

新增 `tests/gem5/npu/vpu_fma/`：

- workload 构造对舍入敏感和不敏感的两类输入
- 至少验证：
  - 正常 3 输入路径
  - `src2` 为 bias
  - `repetition > 1`

### 退出条件

- 三输入访存和结果写回路径稳定。

## 阶段 5：实现 `reduce`

### 目标

加入跨元素聚合算子，为后续 `Softmax` 准备。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 增加 `VREDUCE_SUM`、`VREDUCE_MAX`
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - 支持“输入向量 -> 单值/小向量输出”
  - 明确 reduce 结果写回地址与布局
  - 必要时扩展 `buildMvoutRequests()`，允许只写一个结果 slot

### 测试改动

新增 `tests/gem5/npu/vpu_reduce/`：

- `reduce_sum(int32/fp32)`
- `reduce_max(fp32)`
- 覆盖 `elem_count` 不是 2 的幂的情况
- 覆盖负数与混合正负数

配置脚本继续沿用 `vpu/` 现有模式，只增加结果和统计检查。

### 退出条件

- 规约结果路径稳定。
- `reduce_max` 与 `reduce_sum` 能作为 `Softmax` 的子步骤复用。

## 阶段 6：实现 `VLOAD / VSTORE`

### 目标

提供显式数据搬运语义，使“先 load 到内部 buffer，再算，再 store”成为可能。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 增加 `VLOAD`、`VSTORE`
  - 增加内部 buffer 元数据
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - 实现内部 operand/result buffer 生命周期
  - 允许一条命令只 load，不 execute
  - 允许一条命令只 store，不重新读取源
- [SpecializedExecutionUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/SpecializedExecutionUnit.hh)
  - 如当前 active state 生命周期不够，增加跨命令保留 buffer 的钩子
- [SpecializedExecutionUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/SpecializedExecutionUnit.cc)
  - 如需要，增加完成一条命令后保留部分 VPU 子状态的能力

### 测试改动

新增 `tests/gem5/npu/vpu_loadstore/`：

- case 1: `VLOAD -> VSTORE`，不做计算，验证搬运正确性
- case 2: `VLOAD -> VADD -> VSTORE`
- case 3: 不同 stride / offset
- case 4: 两个 VPU 实例同时操作不同 SPM 区间

注意：

- 如果第一版 `VLOAD/VSTORE` 只定义为 SPM <-> VPU buffer，就不要把 DRAM case 放进这里。
- DRAM 相关数据准备仍然交给 `DmaUnit` 测试。

### 退出条件

- VPU 指令能显式表达“搬运”和“计算”。

## 阶段 7：实现 `Softmax`

### 目标

实现首个真正的复合算子。

### 建议语义

第一版 `Softmax` 固定为：

- 输入：一个 `fp32` 向量
- 步骤：`max` -> `exp(x - max)` -> `sum` -> `divide`
- 输出：同长度 `fp32` 向量

如果当前阶段不想引入 `exp` 通用指令，可先把 `Softmax` 作为单独 opcode，在 `VpuUnit` 内部实现组合流程。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 增加 `VSOFTMAX`
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - 增加 softmax 执行路径
  - 内部复用 `reduce_max`、`reduce_sum` 与浮点逐元素算子 helper

### 测试改动

新增 `tests/gem5/npu/vpu_softmax/`：

- 用小向量做精确或近似比较
- Python 配置和 workload 需要支持“误差阈值判断”，不能继续只做 bit-exact

建议新增测试点：

- 单峰输入
- 全相同输入
- 包含较大正数/负数，验证减 max 后数值稳定
- 长度 1、4、16

### 退出条件

- `Softmax` 数值误差有清晰阈值，并在测试中固定。

## 阶段 8：更一般的浮点运算框架

### 目标

把已经落地的浮点算子抽象成统一框架，便于继续扩展 `min/max/abs/exp/log` 等。

### 需要修改的代码

- [VpuUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.hh)
  - 抽象统一的 unary/binary/ternary FP dispatch helper
- [VpuUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/VpuUnit.cc)
  - 把现有 `fp32` 算子实现收敛到通用模板函数或函数指针分派
  - 把特殊值处理统一到公共 helper
- [src/npu/README.md](/home/wangyukun/workspace/gem5/src/npu/README.md)
  - 更新 VPU 支持的 opcode 和测试状态

### 测试改动

不建议再只在单个 `vpu_*.c` 里堆 case。到这个阶段应补一个共享参考库：

- 新增 `tests/gem5/npu/utils/vpu_ref.hh`
  - 提供 `fp32` unary/binary/reduce/softmax 的软件参考实现

然后把已有 `vpu_elemwise`、`vpu_unary`、`vpu_fma`、`vpu_reduce`、`vpu_softmax` 中重复的参考计算逻辑迁移过去。

### 退出条件

- 后续再加浮点算子，不需要重复写访存和特殊值处理逻辑。

## 5. 每一步建议修改哪些测试

围绕现有 [tests/gem5/npu/vpu](/home/wangyukun/workspace/gem5/tests/gem5/npu/vpu) 建议采用“保留原测试 + 按能力拆新目录”的方式：

- 保留 `vpu/`
  - 继续专门验证现有 mask/repetition/sync/双实例语义
- 新增 `vpu_decode_smoke/`
  - 验证新 opcode 和字段解码
- 新增 `vpu_elemwise/`
  - 覆盖 `VADD/VSUB/VMUL/VDIV`
- 新增 `vpu_unary/`
  - 覆盖 `scale/sqrt/conversion`
- 新增 `vpu_fma/`
  - 覆盖 `VFMA`
- 新增 `vpu_reduce/`
  - 覆盖 `reduce_sum/reduce_max`
- 新增 `vpu_loadstore/`
  - 覆盖 `VLOAD/VSTORE`
- 新增 `vpu_softmax/`
  - 覆盖 `Softmax`

每个目录都沿用现有结构：

- `src/*.c`
- `src/Makefile`
- `configs/*.py`
- `test_*.py`

每个 workload 都应做三件事：

- 初始化 SPM 输入
- 用软件参考模型算期望值
- 轮询结果并打印稳定 PASS marker

每个 config 都应至少检查：

- `exit_cause`
- `exit_code`
- `queueOccupancy() == 0`
- `completedCmdCount()`
- `executeCount()`
- `completedReadRespCount()`
- `completedWriteRespCount()`
- `completedIterationCount()`

对 `Softmax` 这类近似数值测试，建议：

- workload 内做误差判断并打印 PASS/FAIL
- config 只检查退出状态和计数器，不把浮点近似逻辑塞进 Python

## 6. 推荐落地顺序

如果目标是尽快得到可用版本，推荐按下面的提交批次推进：

1. 阶段 0：命令格式与测试 helper
2. 阶段 1：`VADD/VSUB/VMUL`
3. 阶段 2：`scale/conversion/sqrt`
4. 阶段 3：`VDIV`
5. 阶段 4：`VFMA`
6. 阶段 5：`reduce`
7. 阶段 6：`VLOAD/VSTORE`
8. 阶段 7：`Softmax`
9. 阶段 8：浮点通用化与文档整理

这样做的好处是：

- 每一步都能在现有 gem5 测试框架下独立验证。
- 每一步失败时，问题范围都比较小。
- `Softmax` 不会阻塞前面更基础、也更容易验收的算子。

## 7. 风险点

实施时需要重点注意：

- 当前 `SpecializedExecutionUnit` 默认假设“每个 port 对应一个 32-bit slot”；真实向量算子很可能需要把它扩成“每 port 一段连续 buffer”。
- `VLOAD/VSTORE` 与 `DmaUnit` 的职责边界必须先写清楚，否则后续测试会混乱。
- `fp32` 如果直接依赖宿主机浮点，测试必须把舍入、`NaN`、`Inf` 行为先定死。
- `VFMA` 是否真正 fused、`fp32->int32` 如何舍入，这两点必须先定规范，再写测试。
- `Softmax` 需要误差阈值，不适合按位比较。

## 8. 最小可交付版本

如果只追求“先有能用的 VPU 算子集合”，建议最小版本定义为：

- 阶段 0 完成
- 阶段 1 完成
- 阶段 2 完成
- 阶段 3 完成
- 阶段 4 完成

即先交付：

- `VADD`
- `VSUB`
- `VMUL`
- `VDIV`
- `VFMA`
- `scale`
- `sqrt`
- `int/float conversion`

这一版已经能覆盖大部分基础向量算术；`reduce`、`VLOAD/VSTORE`、`Softmax` 再作为第二批功能继续推进。
