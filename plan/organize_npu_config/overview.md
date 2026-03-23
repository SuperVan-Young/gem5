# NPU Test Config 整理计划

## 目标

将 `tests/gem5/npu` 下面分散、重复的 m5 system configuration 收敛成一个统一的 Python class，用它生成 3D DRAM tile NPU 测试架构，并通过内置方法统一配置 NPU 组件，包括 `seu`、`megacmdqueue`、`cpu`、`spm`、`dma` 等。最终所有 testcase 都改为依赖这一套 class，而不是各自手写 `System(...)` 与设备接线。

## 现状分析

### 1. 重复的基础系统骨架

以下配置都重复创建了非常相似的基础系统：

- `System(mem_mode="timing", membus=SystemXBar(), clk_domain=...)`
- `system.system_port = system.membus.cpu_side_ports`
- `RiscvTimingSimpleCPU` 的创建、cache port 连接、`createInterruptController()`
- `SEWorkload.init_compatible(binary)`、`Process(...)`、`createThreads()`
- `Root(...); m5.instantiate()`

典型位置：

- `tests/gem5/npu/tile/configs/tile_megacmdqueue_seu.py:23`
- `tests/gem5/npu/sit/configs/sit_sync_indicator.py:22`
- `tests/gem5/npu/sit/configs/sit_launch_sync_launch_sync.py:21`
- `tests/gem5/npu/seu/configs/seu_basic.py:47`
- `tests/gem5/npu/mega/configs/megacmdqueue_full.py:20`
- `tests/gem5/npu/dma/configs/dma_proto.py:26`
- `tests/gem5/npu/4rv/configs/4rv_sync_stress.py:29`

这些脚本的差异主要不在 system 骨架，而在设备组合、地址布局、process 参数和退出后的检查逻辑。

### 2. 重复的 NPU 设备接线

`MegaCmdQueue` 和 `SpecializedExecutionUnit` 的构造与连接在多个 testcase 中几乎一致，只是参数略有差异：

- `MegaCmdQueue(... base_addr=cmdq_base, range_addr=..., num_sync_indicator=...)`
- `cmdq.cpu_side = system.membus.mem_side_ports`
- `cmdq.sync_indicator_side = system.membus.mem_side_ports`
- `cmdq.mem_side = system.membus.cpu_side_ports`
- `SpecializedExecutionUnit(... macro_cmd_bytes=..., cmd_queue_depth=...)`
- `seu.cpu_side = system.membus.mem_side_ports`
- `seu.mem_side = system.membus.cpu_side_ports`

典型位置：

- `tests/gem5/npu/tile/configs/tile_megacmdqueue_seu.py:45`
- `tests/gem5/npu/sit/configs/sit_sync_indicator.py:44`
- `tests/gem5/npu/sit/configs/sit_launch_sync_launch_sync.py:43`
- `tests/gem5/npu/seu/configs/seu_basic.py:69`
- `tests/gem5/npu/4rv/configs/4rv_sync_stress.py:59`

当前的问题不是“有没有抽 helper”，而是每个 testcase 都重新决定一次：

- 地址常量
- queue depth
- 端口接线
- 是否打开 sync enqueue
- 单核还是多核

这会让后续增加 tile、SEU 或命令通道时，改动扩散到每个 testcase。

### 3. process address map 也在重复分散

`process.map(...)` 目前在 testcase 脚本中手工维护：

- `cmdq_base` 映射在 `tile`、`sit`、`dma`、`mega`
- `sync_base` 映射在 `sit_sync_indicator`、`4rv`
- `spm_base` 和 `dram_base` 映射在 `dma`
- 多核 port base 偏移映射在 `4rv`

这意味着 address layout 的定义和 workload 需要的映射规则没有集中表达，后续一旦 3D DRAM tile NPU 的地址空间调整，容易漏改。

### 4. testcase 已经有“设备组合”趋势，但没有统一模型

从当前目录看，test 已经隐含出几类组合：

- `mega`: 只有 `MegaCmdQueue`
- `seu`: `MegaCmdQueue + SEU`
- `sit`: `MegaCmdQueue + SEU + sync indicator`
- `tile`: 单 tile 的 `MegaCmdQueue + SEU`
- `dma`: `MegaCmdQueue + DMA + SPM + lowmem`
- `4rv`: 多 CPU + `MegaCmdQueue + SEU + sync`

说明合理的抽象层不应该只是“复制一个 common.py”，而应该是：

- 一个统一的 NPU system builder class
- 一组设备启用/配置方法
- 一组 workload / mapping 辅助方法
- 每个 testcase 只声明“我要什么拓扑”和“我要验证什么”

## 建议的重构目标

### 1. 引入统一的 builder class

建议新增一个专门面向测试配置的 Python 模块，例如：

- `tests/gem5/npu/configs/npu_test_system.py`

核心 class 建议命名为：

- `NPUTestSystemBuilder`
- 或 `ThreeDDRAMTileNPUBuilder`

建议优先用 `NPUTestSystemBuilder` 作为对外名字，把 “3D DRAM tile NPU” 作为它默认构造的拓扑语义，而不是把类名绑死在当前一次命名上。这样后面即使拓扑细节扩展，测试 API 也更稳定。

### 2. class 的职责边界

这个 class 不应该承担 testcase 的 pass/fail 判断；它只负责：

- 创建 `System`
- 创建并连接 CPU / CPUs
- 创建并连接 NPU 组件
- 管理统一地址布局
- 为 workload 生成 `Process`
- 统一完成 `process.map(...)`
- 暴露 `build()` / `instantiate()` 后的对象引用，供 testcase 读取统计量

testcase 脚本仍然保留：

- 参数解析
- 选择具体场景
- 调用 builder
- `m5.simulate()` 和结果打印
- verifier 所依赖的最小输出

### 3. 统一的 builder API 草案

建议 class 提供以下几类方法。

基础构造：

- `__init__(clock="1GHz", mem_mode="timing", mem_ranges=None, addr_map=None)`
- `build_base_system()`
- `instantiate_root()`

CPU / workload：

- `add_cpu(cpu_id=0)`
- `add_cpus(num_cpus)`
- `set_workload(binary, argv=None, cpu_id=0, pid=None)`
- `set_workloads(per_cpu_args_fn)`

NPU 设备：

- `add_megacmdqueue(num_input_port, mega_cmd_width, cmd_queue_depth, base_addr, range_addr=None, num_sync_indicator=0)`
- `add_seu(base_addr, macro_cmd_bytes, cmd_queue_depth, debug_process_latency="50ns", sync_enqueue_on_data_write=False)`
- `add_dma(base_addr, macro_cmd_bytes, cmd_queue_depth, buffer_size, sync_enqueue_on_data_write=True)`
- `add_spm(base_addr, size, latency="10ns", bandwidth="100GiB/s")`
- `add_lowmem(size_or_range)`

地址映射：

- `map_cmdq(process=None, cpu_id=0, size=None)`
- `map_sync(process=None, size=4)`
- `map_spm(process=None, base=None, size=None)`
- `map_dram(process=None, base=None, size=None)`
- `map_all_required_regions()`

可观察对象：

- `get_process(cpu_id=0)`
- `get_processes()`
- `system`
- `components["cmdq"]`, `components["seu"]`, `components["dma"]`, `components["spm"]`

### 4. 地址布局应内建成单一事实来源

builder 内建议定义一个默认地址布局对象，例如：

- `CMDQ_BASE = 0x70000000`
- `SYNC_BASE = 0x71000000`
- `SEU_BASE = 0x72000000`
- `CMDQ_RANGE_BASE = 0x73000000`
- `DMA_BASE = 0x74000000`
- `DMA_RANGE_BASE = 0x75000000`
- `SPM_BASE = 0x60000000`
- `DRAM_BASE = 0x20000000`

并支持 testcase 覆盖。重点是：

- 地址不再散落在每个 config 脚本
- `process.map(...)` 从地址布局自动推导
- 多 CPU port base 的偏移规则也集中表达

对于 `4rv` 这种多核场景，建议提供专用 helper：

- `get_cmdq_port_base(cpu_id)`
- `map_cmdq_port(cpu_id, process=None)`

## 推荐的落地结构

建议目录结构如下：

```text
tests/gem5/npu/configs/
  npu_test_system.py
  common.py                # 如果需要放共享常量/地址布局 dataclass
```

不建议把这个 builder 放到 `tests/gem5/npu/utils/`：

- 现有 `utils/` 明显偏 C workload helper
- Python 侧 system config 放在 `configs/` 语义更直接
- testcase config 脚本导入路径也更自然

如果后面 builder 继续增长，可以再拆：

- `npu_addr_map.py`
- `npu_components.py`
- `npu_workload.py`

但第一轮不要拆太细，否则重构本身会先把复杂度抬高。

## 分阶段改写计划

### Phase 0: 基线梳理

目标：

- 确认所有 testcase 当前依赖的 system 形态
- 列清楚每个 testcase 用到的地址、设备、映射、统计量

产出：

- 一张 testcase 到 builder feature 的映射表
- builder 默认地址布局
- builder 最小 API 列表

建议覆盖关系：

- `mega` 需要：单 CPU + `MegaCmdQueue`
- `seu` 需要：单 CPU + `MegaCmdQueue + SEU`
- `sit` 需要：单 CPU + `MegaCmdQueue + SEU + sync map`
- `tile` 需要：单 CPU + tile `MegaCmdQueue + SEU`
- `dma` 需要：单 CPU + `MegaCmdQueue + DMA + SPM + lowmem`
- `4rv` 需要：多 CPU + `MegaCmdQueue + SEU + sync map + per-port mapping`

### Phase 1: 抽出 builder，不改行为

目标：

- 新增 `npu_test_system.py`
- 先把当前重复的 system 构造和设备接线搬进去
- 不改变任何 testcase 的功能语义

实施要求：

- builder 的默认值必须对齐现有脚本行为
- 先支持单 CPU 和多 CPU
- 先支持 `MegaCmdQueue`、`SEU`、`DMA`、`SPM`、`lowmem`
- 先把 `instantiate` 前的组装流程稳定下来

这一阶段不要做的事：

- 不重写 verifier
- 不调整输出格式
- 不顺手修改 workload C 程序
- 不引入额外“聪明”的自动推断逻辑

### Phase 2: 统一 process/workload 映射接口

目标：

- 把 `Process(...)`、`process.cmd`、`cpu.createThreads()`、`process.map(...)` 纳入 builder
- testcase 不再直接写地址映射细节

实施重点：

- 支持单进程单核
- 支持多核多进程
- 支持为每个 CPU 传不同 argv
- 支持 testcase 声明“我要 cmdq/sync/spm/dram 映射”，而不是自己算地址

预期结果：

- `4rv_sync_stress.py` 的多核配置代码会明显缩短
- `sit` / `tile` / `dma` 会只保留场景参数与结果检查

### Phase 3: 全量迁移 testcase

建议迁移顺序：

1. `mega`
2. `spm`
3. `seu`
4. `sit`
5. `tile`
6. `dma`
7. `4rv`

原因：

- `mega`、`spm` 最简单，适合先验证 builder 基础骨架
- `seu`、`sit`、`tile` 能验证共享的 `MegaCmdQueue + SEU` 路径
- `dma` 引入特殊 memory layout
- `4rv` 最后迁移，因为它包含多 CPU 和 per-port mapping，是复杂度最高的场景

每迁移一个 testcase，都应检查：

- 输出 pass marker 不变
- verifier 不需要同步改动，或只做最小改动
- 设备统计量读取路径不变或可平滑替换

### Phase 4: 清理和收口

目标：

- 删除迁移后不再需要的重复配置代码
- 统一 config 文件风格
- 补充 builder 的最小文档注释

可以额外做的轻量清理：

- 统一 `expected_exit_cause` 常量命名
- 统一 `cmd_width` / `cmd_bytes` 的计算位置
- 统一 `binary = os.path.abspath(args.binary)` 与 workload 创建方式

## 每个 testcase 的改写形态

重构后的 testcase config 脚本建议收敛成以下结构：

1. 解析参数
2. 创建 builder
3. 调用若干 `add_*` / `set_*` 方法描述拓扑
4. 调用统一的 mapping helper
5. `instantiate`
6. `simulate`
7. 读取 builder 暴露出的组件统计量
8. 打印 pass marker

也就是说，testcase 负责“声明场景”，builder 负责“搭系统”。

## 风险与注意事项

### 1. 不要把 testcase 差异过度抽象掉

`seu_basic.py` 当前使用轮询 `simulate(poll_step_ticks)`，而其余大多是一次 `m5.simulate()`。builder 不应强行统一运行流程，否则会把“系统搭建”和“测试控制流”耦合起来。

### 2. 多核 cmdq 映射规则必须显式保留

`4rv_sync_stress.py` 当前通过 `cmdq_base + (cpu_id << 20)` 计算 port base。这个规则不能在迁移时被隐式吞掉，必须在 builder API 中显式体现。

### 3. DMA 的 memory layout 不能被 512MiB 默认值覆盖

`dma_proto.py` 使用的是 `lowmem + spm` 的组合 memory range，而不是普通的 `physmem=SimpleMemory(range=AddrRange("512MiB"))`。builder 必须允许替换默认 memory 拓扑，而不是所有 case 共用一个 `physmem` 模板。

### 4. class 的默认参数要保守

builder 默认值应尽量匹配当前最常见场景，但不要让 testcase 因为“没写参数”而悄悄获得额外设备或映射。

## 验收标准

完成这轮改写后，应满足：

- `tests/gem5/npu` 下所有 testcase config 都通过统一 builder/class 创建 system
- testcase 中不再重复手写 CPU 基础接线、`MegaCmdQueue` 接线、`SEU` 接线
- 地址布局在单一模块中维护
- testcase 只保留场景参数、少量特定映射选择和结果检查
- 多核、DMA、SPM 这些特殊场景都能由同一套 builder API 覆盖

## 建议的第一步实现

如果下一步开始真正改代码，建议先做一个最小可用版本：

1. 新建 `tests/gem5/npu/configs/npu_test_system.py`
2. 实现基础 system、单/多 CPU、`MegaCmdQueue`、`SEU`
3. 先迁移 `mega`、`seu`、`sit_sync_indicator`
4. 等 API 稳定后，再接入 `tile`、`dma`、`4rv`

这样可以先把最常见的重复去掉，再逐步吸收复杂场景，避免第一次就把多核、DMA、SPM、3D DRAM tile 全部绑死在一个未经验证的大 class 里。
