内容由 AI 生成

一、详细版 agent.md

你可以直接把下面内容保存成 agent.md：

# 双 VPU 实现任务说明（agent.md）

## 0. 文档目的

本文档用于指导 agent 在一个基于 gem5 的 NPU 仿真工程中实现"双 VPU"功能，并补齐完整测试。

本文档的目标不是让 agent 自由发挥，而是：

1. 明确当前代码事实；
2. 明确推荐实现路线；
3. 明确哪些改动可以直接做，哪些改动必须先向用户确认；
4. 明确如何拆分任务并选择模型；
5. 明确测试、验收、日志、失败处理和最终交付格式。

本轮追求的是：**双 VPU 最小可行闭环**。  
不是重写整个 SEU 架构，也不是一次性做完未来所有功能单元。

---

## 1. 背景与总体目标

SEU 部分的长期目标是包含 5 个功能单元：

- `VPU0`
- `VPU1`
- `MPU`
- `DMA`
- `LUT`

这五个单元都连在同一套总线抽象上。当前工程抽象中，数据与控制信号共用一条总线。

### 1.1 本轮只做什么

本轮只要求实现：

- 两个 VPU 单元；
- 测试系统中的双 VPU 实例接入；
- 通过 `MegaCmdQueue` 正确寻址到对应 VPU；
- 软件侧测试程序；
- 配套 gem5 测试脚本与验收输出。

### 1.2 本轮不要求做什么

本轮**不要求**：

- 完成 MPU / DMA / LUT 的新功能语义；
- 新增统一总线仲裁器；
- 新增片内互连重构；
- 新增时序争用建模；
- 追求"论文级完整架构"；
- 发明新的复杂 VPU ISA；
- 为了"更优雅"而大范围重构已有代码。

如果需要上述范围外改动，必须先形成改动说明并向用户确认。

---

## 2. 当前代码事实（必须基于这些事实做决策）

### 2.1 MegaCmdQueue 已有寻址能力

当前 `src/npu/mega/MegaCmdQueue.cc` 已按 `deviceType/deviceId` 计算目标地址：

- `target = 0x70000000 | (deviceType << 24) | (deviceId << 20)`

因此如果约定：

- `deviceType = 0x2` 表示 VPU

那么：

- `deviceId = 0` 会映射到 `0x72000000`
- `deviceId = 1` 会映射到 `0x72100000`

这意味着双 VPU 可以直接复用现有命令头分发机制，不需要先重构 `MegaCmdQueue` 协议。

### 2.2 SpecializedExecutionUnit 已有可复用执行框架

当前 `src/npu/mega/SpecializedExecutionUnit.{hh,cc,py}` 是通用执行单元基类，已经支持：

- MMIO staging / launch
- 内部命令队列
- 单命令状态机
- `readMask / writeMask / repetition`
- 多 `mem_side` 端口
- 端口级 read / execute / write 的 toy 流程

因此，双 VPU 的最小风险路线应优先是：

- 新增 `VpuUnit`，继承 `SpecializedExecutionUnit`
- 尽量复用基类已有执行框架
- 在 `VpuUnit` 层收敛 VPU 语义，而不是破坏基类

### 2.3 DmaUnit 是派生单元风格参考

`src/npu/mega/DmaUnit.{hh,cc,py}` 已经给出"从 `SpecializedExecutionUnit` 派生出真实功能单元"的风格参考。

agent 在写 `VpuUnit` 时，应优先对齐这类派生风格，而不是自行创造一套完全不同的组织方式。

### 2.4 测试系统当前只有单 SEU 辅助接口

`tests/gem5/npu/configs/npu_test_system.py` 当前只提供：

- `add_seu()`
- `map_seu()`

因此双 VPU 需要在不破坏旧测试的前提下，新增：

- `add_vpu()`
- `map_vpu()`

不要删除、重命名、重载式破坏现有 `add_seu()` / `map_seu()` 行为。

### 2.5 现有旧用例默认行为不能破坏

现有 `SEU` / `tile` / `sit` / `4rv` 等测试路径默认使用 `deviceId=0`。  
本轮新实现必须尽量局部化，避免影响旧路径。

### 2.6 用户指定目标验证命令

目标验证命令为：

```bash
scons build/RISCV/gem5.opt -j8
./build/RISCV/gem5.opt --debug-flags=VPU tests/gem5/npu/vpu/configs/vpu_basic.py --binary tests/gem5/npu/vpu/bin/vpu_test_riscv
```

agent 必须尽量以这两条命令为最终验证基准。

---

## 3. 本轮推荐实现路线

### 3.1 总体原则

不要把"SEU 内两个 VPU"做成一个对象内部的双调度器。

当前代码基线已经把"单元实例"抽象为独立 SimObject，且 MegaCmdQueue 已经按 deviceId 做地址分发，因此最小风险方案是：

- 新增 VpuUnit 类；
- 继承 SpecializedExecutionUnit；
- 在系统里挂两个 VpuUnit 实例；
- 由 MegaCmdQueue 根据命令头中的 deviceId=0/1 自动寻址到 vpu0/vpu1；
- 在测试中显式构造分别发往两个 VPU 的命令并验证行为。

### 3.2 推荐实例布局

建议约定：

- deviceType = 0x2
- vpu0.deviceId = 0
- vpu1.deviceId = 1

推荐地址：

- vpu0 基地址：0x72000000
- vpu1 基地址：0x72100000

在 npu_test_system.py 中推荐默认写法：

```python
base_addr = addr_map.seu_base + (vpu_id << 20)
```

这里沿用现有地址图命名，不代表 VPU 与旧 SEU 是同一个对象。

### 3.3 为什么不建议做单对象双调度

因为那会明显增加：

- 内部状态耦合风险
- 命令归属错误风险
- 统计值串扰风险
- 调试复杂度
- 与现有 MegaCmdQueue 自然寻址模型的不匹配

本轮目标是先把双 VPU 最小闭环跑通，而不是追求"看起来更集中"的内部设计。

---

## 4. 任务拆分与模型分配规则

agent 必须先拆任务，再执行。

### 4.1 基本分配原则

#### 适合交给 GPT-5.3 的任务

用于简单、局部、低风险、机械性较强的工作，例如：

- 新测试目录结构搭建
- Makefile 补齐
- 小型 SConscript 衔接
- 简单 Python glue 代码
- 日志打印模板
- 重复性样板代码
- 不涉及关键架构边界的纯搬运式修改

#### 适合交给 GPT-5.4 的任务

用于复杂、跨文件、强耦合、需要架构判断的工作，例如：

- VpuUnit 的设计与实现
- 与 SpecializedExecutionUnit 的继承边界处理
- 双 VPU 的状态隔离设计
- npu_test_system.py 中双 VPU 的接入策略
- 测试 oracle 设计
- 编译失败 / 运行失败的定位
- 集成验收与最终 correctness 判断

### 4.2 分配时必须遵守的原则

- 先做任务拆分，再做实现；
- 每个子任务都要解释为什么交给 GPT-5.3 或 GPT-5.4；
- 如果某项任务看起来简单，但一旦做错会污染公共路径或破坏现有用例，则应视为复杂任务，交给 GPT-5.4；
- 最终回复中必须回顾整个分解与模型分配结果。

---

## 5. 本轮可以直接修改的文件范围

本轮优先允许修改：

### 5.1 源码目录

- `src/npu/mega/VpuUnit.hh`
- `src/npu/mega/VpuUnit.cc`
- `src/npu/mega/VpuUnit.py`
- `src/npu/mega/SConscript`

### 5.2 测试系统与测试目录

- `tests/gem5/npu/configs/npu_test_system.py`
- `tests/gem5/npu/vpu/configs/vpu_basic.py`
- `tests/gem5/npu/vpu/src/vpu_test_riscv.c`
- `tests/gem5/npu/vpu/src/Makefile`
- `tests/gem5/npu/vpu/test_vpu.py`

### 5.3 可酌情新增但应尽量少的文件

- `tests/gem5/npu/vpu/.gitignore`
- 构建或测试所必需的少量辅助文件

**原则：** 只做实现目标所必需的最小增量，不随意扩展修改面。

---

## 6. 改动前必须先请求用户确认的事项

下面这些改动不是绝对禁止，但在真正修改前，必须先整理清单并向用户确认：

1. 修改 MegaCmdQueue 的命令头协议、地址分发规则或 MMIO 布局
2. 将双 VPU 设计为"单对象内部双调度器"
3. 重命名、删除、拆散或大改 SpecializedExecutionUnit
4. 删除、重写或改变现有 add_seu() / map_seu() 的语义
5. 影响现有 SEU / tile / sit / 4rv 测试公共路径的改动
6. 引入新的命令格式、新软件接口或新的 VPU ISA 语义
7. 引入总线仲裁器、片内互连重构、时序争用建模等大改动
8. 任何"更优雅但非必要"的重构

**请求确认时，必须明确写出：**

- 为什么现有最小方案不够
- 需要改哪些文件
- 改动收益是什么
- 对旧测试可能有什么影响
- 为什么不能用更小改动完成

在用户确认前，不得先做这些改动。

---

## 7. 双 VPU 的精确定义

### 7.1 实例要求

双 VPU 必须是两个独立实例：

- vpu0
- vpu1

每个实例必须具备：

- 自己的命令队列
- 自己的执行状态机
- 自己的统计计数
- 自己的 debug 打印上下文

### 7.2 命令归属要求

- deviceId = 0 的命令必须只被 vpu0 消费
- deviceId = 1 的命令必须只被 vpu1 消费
- 任一 VPU 不得消费属于另一 VPU 的命令
- 不能出现 completed 计数、内部缓冲结果、状态机阶段串扰

### 7.3 本轮最小语义

由于用户当前只明确要求"实现两个 VPU 的功能"，并未给出更细的 VPU 算法规范，因此本轮采用与 SpecializedExecutionUnit 当前默认 toy 语义一致的最小可行方案：

- 只支持 opCode = EXEC(0x0)
- readMask 表示读哪些 slot
- writeMask 表示写哪些 slot
- repetition 表示重复迭代次数
- 写回值沿用当前默认公式
- 每个 slot / port 的处理流程尽量复用现有基类机制

### 7.4 对不支持命令的处理

如果收到本轮不支持的 opCode：

- 可以明确报错
- 可以拒绝执行
- 可以 panic/fatal

**但不能静默忽略。**

### 7.5 deviceId 检查

若命令里的 deviceId 与当前实例不匹配，应明确拒绝执行并输出足够的 debug 信息。

### 7.6 两个 VPU 的独立性

双 VPU 之间：

- 不共享命令状态
- 不共享统计值
- 不共享内部暂存结果
- 不共享"当前命令"上下文

两者唯一可能的外部相关性，只能来自共享 SPM 的最终数据可见效果。

---

## 8. 必须完成的具体代码改动

### 8.1 src/npu/mega 中新增 VpuUnit

必须新增：

- `VpuUnit.hh`
- `VpuUnit.cc`
- `VpuUnit.py`

**要求：**

- 继承 SpecializedExecutionUnit
- 可作为独立 SimObject 实例化两次
- 接入 SCons 构建系统
- 有自己的 debug flag，例如 VPU
- 风格尽量贴近 DmaUnit / SpecializedExecutionUnit

### 8.2 更新构建系统

更新：`src/npu/mega/SConscript`

确保 VpuUnit：

- 被正确编译
- 被正确导出到 Python SimObject
- 能在测试系统配置脚本中 import

### 8.3 修改测试系统封装

修改：`tests/gem5/npu/configs/npu_test_system.py`

**要求新增：**

```python
from m5.objects import VpuUnit
add_vpu(...)
map_vpu(...)
```

**推荐接口：**

```python
add_vpu(vpu_id, attr_name=None, base_addr=None, num_mem_side_ports=...)
```

**推荐默认值：**

```python
base_addr = addr_map.seu_base + (vpu_id << 20)
attr_name = f"vpu{vpu_id}"
```

**注意：**

- 不要删除现有 add_seu() / map_seu()
- 不要通过改旧接口来"兼容双 VPU"
- 新旧路径应尽量并存

### 8.4 新增双 VPU 测试目录

新增：

- `tests/gem5/npu/vpu/configs/vpu_basic.py`
- `tests/gem5/npu/vpu/src/vpu_test_riscv.c`
- `tests/gem5/npu/vpu/src/Makefile`
- `tests/gem5/npu/vpu/test_vpu.py`

如仓库规范需要，可补：

- `tests/gem5/npu/vpu/.gitignore`

---

## 9. 测试要求

### 9.1 vpu_basic 的建议系统配置

建议最小系统包含：

1. 1 个 CPU
2. 1 个 MegaCmdQueue
3. 1 个 ScratchpadMemory
4. 2 个 VpuUnit
5. 每个 VpuUnit 先设 num_mem_side_ports = 4

**原则：**

- 先保证简单、稳定、可调试
- 不要一开始就引入复杂并发
- 不要把失败原因复杂化

### 9.2 测试阶段一：基础寻址验证

必须至少发出两条命令：

- 一条发往 deviceId=0
- 一条发往 deviceId=1

**要求：**

- 两条命令使用不同的 readMask
- 两条命令使用不同的 writeMask
- 两条命令使用不同的 sync idx
- 两条命令访问不同或可区分的 SPM 区域

**验证目标：**

- vpu0 只影响属于 vpu0 的预期结果
- vpu1 只影响属于 vpu1 的预期结果

### 9.3 测试阶段二：独立状态验证

在基础寻址验证后，再发若干条命令，混合顺序提交到 MegaCmdQueue。

至少检查：

```python
builder.system.vpu0.completedCmdCount()
builder.system.vpu1.completedCmdCount()
```

并验证：

- 两边计数独立正确
- 两边统计互不串扰

### 9.4 测试阶段三：结果正确性验证

vpu_test_riscv.c 中必须在软件侧自己计算 expected 值，并与实际 SPM 结果逐项比较。

**不允许只验证：**

- 程序没崩
- 命令发出去了
- 日志里像是执行了

**通过条件必须基于：**

- SPM 结果正确
- 双 VPU 完成计数正确
- 最终打印 VPU_TEST_PASS

### 9.5 日志可诊断性要求

DPRINTF(VPU, ...) 至少应覆盖：

- 命令头关键信息
- deviceId
- opCode
- readMask
- writeMask
- repetition
- 每轮 iteration
- 关键读请求
- 关键写请求
- 完成同步信息

这样当失败时，能判断是：

- 命令没送到正确实例
- 状态机没推进
- 读写链路不匹配
- 统计值错了
- 软件 expected 算错了

---

## 10. 建议实施顺序

推荐按以下顺序执行：

1. 分解子任务，并决定 GPT-5.3 / GPT-5.4 分工
2. 新建 VpuUnit 类
3. 接入 SConscript / Python SimObject / debug flag
4. 修改 npu_test_system.py 增加 add_vpu() / map_vpu()
5. 编写 vpu_test_riscv.c，先只测两条命令分别发到 vpu0 / vpu1
6. 编写 vpu_basic.py，打印：
   - exit cause
   - vpu0 / vpu1 的 completed / prologue / execute / epilogue / read_resp / write_resp / iterations
   - 关键 SPM slot 值
   - VPU_TEST_PASS
7. 编写 test_vpu.py
8. 编译 RISC-V 测试二进制, 用workspace/gem5/tests/gem5/npu/spm/src/Makefile里的编译器

**运行：**

```bash
scons build/RISCV/gem5.opt -j8
./build/RISCV/gem5.opt --debug-flags=VPU tests/gem5/npu/vpu/configs/vpu_basic.py --binary tests/gem5/npu/vpu/bin/vpu_test_riscv
```

若失败，再联合：

```bash
./build/RISCV/gem5.opt --debug-flags=VPU,MegaCmdQueue tests/gem5/npu/vpu/configs/vpu_basic.py --binary tests/gem5/npu/vpu/bin/vpu_test_riscv
```

---

## 11. 验收标准

只有同时满足以下条件，才算完成：

1. VpuUnit 成功接入构建系统
2. gem5 编译成功
3. 测试二进制成功生成
4. 两个 VPU 都能被正确寻址
5. 两个 VPU 的状态与统计彼此独立
6. SPM 结果与软件 expected 完全一致
7. 测试输出 VPU_TEST_PASS
8. 不破坏现有测试基线
9. 最终交付中给出真实编译与运行结果，而非假设性结果

---

## 12. 失败时的处理原则

若未完全成功，必须：

- 如实报告失败点
- 明确失败发生在：
  - 编译阶段
  - 链接阶段
  - SimObject 导出阶段
  - 运行阶段
  - 测试判定阶段
- 给出最小修复建议
- **不得通过删除检查项、弱化测试、屏蔽日志、降低标准来伪造通过**

---

## 13. 最终交付格式要求

最终回复必须包含：

1. 子任务分解结果
2. 每个子任务分配给 GPT-5.3 或 GPT-5.4 的理由
3. 总体设计说明
4. 修改文件列表
5. 每个文件的完整内容
6. 编译命令
7. 运行命令
8. 实际结果
9. 若未完全成功，剩余问题与下一步建议
