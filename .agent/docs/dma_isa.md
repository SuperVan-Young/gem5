# DMA 模块宏指令 ISA / Live Reference (DMA 2.5, 512-bit)

## 1. 目的与文档关系

本文档是当前仓库 `src/npu/DmaUnit.*` 的 **live implementation-aligned**
DMA ISA / reference 文档，用于统一描述：

- `DmaUnit` 当前实际解析的命令格式
- `DmaUnit` 当前实际执行/校验的 DMA 语义
- DMA-local workload / testcase 当前应生成的合法命令

当前文档关系：

- `.agent/plan/dma2.1_plan_v1.md` 仍然保留为 DMA 2.1 的控制性计划文档
- 本文档更新为 **DMA 2.5 当前实现现实** 的 reference 文档
- 若计划文档与当前实现存在差异，计划文档并未在 DMA 2.5 中同步更新
- 因此，涉及 `CommandStage`、staged `move_layout`、DMA-local overlap 的
  当前事实，应以本文件和 live code 为准

本文件不是开发日记；它描述的是 **当前代码已实现且已验证** 的 DMA 语义边界。

---

## 2. 当前 DMA 2.5 边界

DMA 2.5 延续 DMA 2.1 的大部分冻结约束：

- 所有 DMA 模式仍然 **INT8-only**
- `move_layout` / `transpose` / `fill` 三种模式仍然保留
- `bank_size` 仍是 DMA-local workspace capacity 的权威控制参数
- 外部 `DRAM` / `SPM` fill 仍必须保持 `reads = 0`
- transpose 仍要求 `src_k == 0 && dst_k == 0`
- 不引入 shared `SEU` / `MegaCmdQueue` 改造

DMA 2.5 在此基础上新增并落地的是：

- `op_code[1:0]` 当前由 live `DmaUnit` 解释为 `CommandStage`
- staged 语义当前只对 `move_layout` 生效
- DMA-local `move_layout` staged 命令支持 batch-level overlap

DMA 2.5 明确仍 **不包含**：

- staged `fill`
- staged `transpose`
- line-level overlap
- streaming overlap
- shared scheduler / shared queue behavior changes
- 新的显式 public bank-selection 合约

---

## 3. 基础假设

### 3.1 地址空间

- DRAM: `0x2000_0000` ~ `0x5FFF_FFFF`
- SPM: `0x6000_0000` ~ `0x6FFF_FFFF`
- DMA 宏指令中的外部地址字段仍按 32-bit 物理地址编码
- `DMA_BANK` 是 DMA internal workspace，不是外部物理地址空间
- `Word 1` / `Word 2` 仍表示外部 `DRAM` / `SPM` 地址字段
- 当 `fill` 目标是 `DMA_BANK` 时，`dst_base_addr` 必须为 `0`

### 3.2 字编号约定

- `Word 0 = bits <31:0>`
- `Word 1 = bits <63:32>`
- ...
- `Word 15 = bits <511:480>`

### 3.3 通用访存与布局约束

- DMA 外部访存仍以 64B cache line 为基本读写单位
- `shape + stride + k` 仍是 layout 描述机制
- 当前所有 DMA 模式均为 `INT8-only`
- `data_type = 000` 是唯一合法编码
- `move_layout` 继续使用显式 cut-dimension 解释 `src_k` / `dst_k`
- `transpose` 扩展后仍要求 `src_k == 0 && dst_k == 0`
- `fill` 对外部 `DRAM` / `SPM` 的实现仍必须保持 `reads = 0`
- DMA 2.5 仍不允许 external fill 的 read-before-write / read-modify-write
- DMA 2.5 的 move-layout batching/workspace 仍以 bank-local 语义实现

---

## 4. 512-bit 宏指令布局

| Word | 字段 | 位段 | 含义 |
| :--- | :--- | :--- | :--- |
| `Word 0` | `Common_Header` | `<31:0>` | 公共头部；DMA-specific `op_code` 解释见下文 |
| `Word 1` | `src_base_addr` | `<63:32>` | 外部源地址；`fill` 中必须为 `0` |
| `Word 2` | `dst_base_addr` | `<95:64>` | 外部目的地址；`fill -> DMA_BANK` 时必须为 `0` |
| `Word 3` | `tensor_shape_h` | `<127:96>` | 逻辑 H |
| `Word 4` | `tensor_shape_w` | `<159:128>` | 逻辑 W |
| `Word 5` | `tensor_shape_c` | `<191:160>` | 逻辑 C |
| `Word 6` | `src_stride_h` | `<223:192>` | 源 H stride；`fill` 中必须为 `0` |
| `Word 7` | `src_stride_w` | `<255:224>` | 源 W stride；`fill` 中必须为 `0` |
| `Word 8` | `src_stride_c` | `<287:256>` | 源 C stride；`fill` 中必须为 `0` |
| `Word 9` | `dst_stride_h` | `<319:288>` | 目的 H stride |
| `Word 10` | `dst_stride_w` | `<351:320>` | 目的 W stride |
| `Word 11` | `dst_stride_c` | `<383:352>` | 目的 C stride |
| `Word 12` | `dma_block_cfg` | `<415:384>` | `[31:16] dst_k`, `[15:0] src_k`；`fill` 中必须为 `0` |
| `Word 13` | `mode_cfg` | `<447:416>` | mode-specific 控制字 |
| `Word 14` | `bank_cfg` | `<479:448>` | bank-specific 控制字 |
| `Word 15` | `fill_value_or_reserved` | `<511:480>` | `fill` 中低 8 位为 `fill_value_int8`；其他模式必须为 `0` |

---

## 5. Word 0: DMA-specific opcode 解释

`Word 0` 仍沿用公共头部格式：

- `device_type[31:28] = 0b0100` (DMA)
- `device_id[27:24]`
- `op_code[23:16]`
- `sync_indicator[15:8]`
- `set_indicator_sns[7]`
- `set_indicator_snd[6]`
- 其余保留

当前 live `DmaUnit` 对 `op_code[23:16]` 的解释为：

- `op_code[7:5] = data_type`
- `op_code[4:2] = mode`
- `op_code[1:0] = command_stage`

### 5.1 data_type

- `000 = INT8`
- 其他编码保留并必须被拒绝

### 5.2 mode

- `000 = move_layout`
- `001 = transpose`
- `010 = fill`
- `011..111 = reserved`

### 5.3 CommandStage

当前 `CommandStage` 编码：

- `00 = Legacy`
- `01 = Load`
- `10 = Compute`
- `11 = Store`

语义说明：

- `Legacy`
  - 保持原有单条宏指令的完整 DMA pipeline 语义
  - 对现有 legacy `move_layout` / `transpose` / `fill` 兼容
- `Load`
  - 仅对 staged `move_layout` 生效
  - 仅执行下一 batch 的 source/destination line reads
- `Compute`
  - 仅对 staged `move_layout` 生效
  - 仅执行下一 loaded batch 的 layout transform
- `Store`
  - 仅对 staged `move_layout` 生效
  - 仅执行下一 computed batch 的 external writes

### 5.4 CommandStage 的当前边界

- staged 语义当前只对 `move_layout` 实现
- staged `fill` 必须被拒绝
- staged `transpose` 必须被拒绝
- `move_layout` staged 路径仍要求 `bank_cfg == 0`
- 当前 staged 路径不暴露新的 public bank-selection 字段

### 5.5 当前为什么 2 bits 足够

对 **当前 DMA 2.5 scope**，2 bits 足够，因为 live 实现只需要四种编码：

- `Legacy`：保留完整兼容路径
- `Load`：staged move_layout 读阶段
- `Compute`：staged move_layout 计算阶段
- `Store`：staged move_layout 写阶段

这四种值已经完全覆盖当前 DMA 2.5 的 staged 需求，因此：

- 对当前 DMA-local staged `move_layout`
- 对当前 batch-level overlap
- 对当前拒绝 staged `fill` / staged `transpose`

2 bits 都是足够的。

但这 **只对当前 DMA 2.5 scope 成立**。未来若要加入：

- 额外 staged modes
- 更细粒度的 runtime phases
- staged barriers / fences / token semantics
- 非 `move_layout` 的独立 staged state machine

则不能自动假设 2 bits 仍然足够，必须重新评估并重新冻结文档。

---

## 6. Word 13 = mode_cfg

### 6.1 move_layout

- `[0] src_mem_space`
  - `0 = DRAM`
  - `1 = SPM`
- `[1] dst_mem_space`
  - `0 = DRAM`
  - `1 = SPM`
- `[3:2] src_cut_dim`
  - `0 = H`
  - `1 = W`
  - `2 = C`
  - `3 = reserved`
- `[5:4] dst_cut_dim`
  - `0 = H`
  - `1 = W`
  - `2 = C`
  - `3 = reserved`
- `[31:6] reserved = 0`

### 6.2 transpose

- `[0] src_mem_space`
  - `0 = DRAM`
  - `1 = SPM`
- `[1] dst_mem_space`
  - `0 = DRAM`
  - `1 = SPM`
- `[3:2] transpose_dim_a`
  - `0 = H`
  - `1 = W`
  - `2 = C`
  - `3 = reserved`
- `[5:4] transpose_dim_b`
  - `0 = H`
  - `1 = W`
  - `2 = C`
  - `3 = reserved`
- `[31:6] reserved = 0`

### 6.3 fill

- `[1:0] dst_mem_space`
  - `0 = DRAM`
  - `1 = SPM`
  - `2 = DMA_BANK`
  - `3 = reserved`
- `[31:2] reserved = 0`

---

## 7. Word 14 = bank_cfg

### 7.1 move_layout

- `bank_cfg` 必须为 `0`
- legacy `move_layout` 和 staged `move_layout` 都不暴露显式 bank 选择字段
- staged `move_layout` 的内部 batch slots / workspace 仍然是 DMA-local 实现细节

### 7.2 transpose

- `[3:0] src_bank_id`
- `[7:4] dst_bank_id`
- `[31:8] reserved = 0`

规则：

- `src_bank_id < num_banks`
- `dst_bank_id < num_banks`
- `src_bank_id != dst_bank_id`

### 7.3 fill

- `[3:0] dst_bank_id`
- `[31:4] reserved = 0`

规则：

- 若 `dst_mem_space = DMA_BANK`，则 `dst_bank_id < num_banks`
- 若 `dst_mem_space = DRAM` 或 `SPM`，则 `dst_bank_id == 0`

---

## 8. Word 15 = fill_value / reserved

### 8.1 move_layout / transpose

- `Word 15 == 0`
- 不得被消费

### 8.2 fill

- `Word 15[7:0] = fill_value_int8`
- `Word 15[31:8] = 0`
- `fill=0` 可直接表示为 `Word 15 = 0`

当前实现仍不定义 `INT16` / `INT32` fill-value 解释。

---

## 9. Per-mode 参考语义

## 9.1 move_layout

### 外部端点

- 仅允许外部 `DRAM` / `SPM` 端点
- `src_mem_space` / `dst_mem_space` 必须与 `src_base_addr` /
  `dst_base_addr` 的实际地址范围一致

### layout 语义

- `shape + stride + k` 仍是 layout 描述机制
- `src_k` 按 `src_cut_dim` 解释
- `dst_k` 按 `dst_cut_dim` 解释
- blocked-layout 合法性必须针对选中的 cut-dimension 显式检查
- 非 canonical blocked-layout 不允许静默猜测解释

### Legacy move_layout

当 `command_stage = Legacy` 时：

- 保持原有完整 DMA pipeline 行为
- 单条命令内部仍按完整 `read -> compute -> write` 批处理执行
- 当前已存在的 legacy testcase / workload helper 继续依赖该路径

### staged move_layout

当 `command_stage != Legacy` 时：

- staged 语义只对 `move_layout` 生效
- 一个 active staged sequence 必须从 `Load` 命令开始
- 后续 `Compute` / `Store` 命令必须与该 sequence 的
  `move_layout` descriptor 完全匹配
- `stage` / sync bits 之外的 descriptor 不允许变化
- active staged sequence 存在时，不允许 interleave 其他非 staged DMA 宏命令

当前 live 实现的 sequence 规则：

- 最多同时存在一个 active staged `move_layout` sequence
- sequence 维护一组共享 `iterationPlans`
- `Load` / `Compute` / `Store` 各自登记自己的 `macroCmdId`
- sequence 结束条件是：
  - 三个 staged 宏命令都已经结束
  - 且所有 planned batches 都已经 store 完成

### staged runtime 限制

对 staged `move_layout`：

- `Load` 命令只构建 read uops
- `Compute` 命令只构建 exec uops
- `Store` 命令只构建 write uops

当前 live 代码中的 issue 分类为：

- `Load -> MacroCmdKind::Load -> read issue queue`
- `Compute -> MacroCmdKind::Exec -> exec queue`
- `Store -> MacroCmdKind::Store -> write issue queue`

### overlap 语义

当前 DMA 2.5 overlap 的含义是：

- 仅限 staged `move_layout`
- 仅限 batch-level overlap
- 允许 `Store(N)` 与 `Load(N+1)` 在当前 SEU 模型上重叠
- 不改变 command-level in-order completion
- 不实现 line-level / streaming overlap
- 不引入 shared `SEU` / `MegaCmdQueue` 行为修改

### 正确性保护

当前实现通过 DMA-local hazard/control 逻辑维持正确性：

- `Load(N+1)` 只有在 load slot 空闲时才能继续
- `Compute(N)` 只有在 `Load(N)` 完成且 write slot 空闲时才能继续
- `Store(N)` 只有在 `Compute(N)` 完成后才能继续
- 若 `Load(N+1)` 的 destination lines 与待写回的 `Store(N)` destination
  lines 冲突，则必须阻塞该次 overlap 尝试

该 hazard gate 当前通过目的 cache-line 地址比较实现，
不是 shared scheduler 机制。

### overlap 观测

当前 overlap 的主观测面不是 debug 文本，而是 DMA-local counter：

- `observedBatchOverlapCount()`

该计数器在下列条件满足时递增：

- active mem transactions 中，存在 `Store(N)` 与 `Load(N+1)` 同时在飞
- 且属于同一 active staged sequence

可选 secondary observability：

- debug 输出 `DMA_OVERLAP_OBSERVE ...`

---

## 9.2 transpose

- transpose 当前只支持 `Legacy`
- staged `transpose` 必须被拒绝
- 拒绝行为是显式 panic，而不是退化到 legacy 路径
- transpose 仍要求：
  - `src_k == 0`
  - `dst_k == 0`
  - `src_bank_id != dst_bank_id`
  - `bank_cfg` 只使用当前 transpose bank 字段

---

## 9.3 fill

- fill 当前只支持 `Legacy`
- staged `fill` 必须被拒绝
- 拒绝行为是显式 panic，而不是退化到 legacy 路径
- external fill 仍必须保持 `reads = 0`
- `DMA_BANK` fill 仍是 DMA-local 行为

---

## 10. 当前实现边界汇总

当前 live DMA 2.5 可以安全假设的事实：

- `op_code[1:0]` 当前不是 reserved，而是 `CommandStage`
- `CommandStage` 当前只有四个值：`Legacy / Load / Compute / Store`
- staged 语义当前只为 `move_layout` 实现
- staged overlap 当前只为 `move_layout` 提供
- overlap 当前只到 batch-level
- `bank_cfg == 0` 仍是 staged `move_layout` 的 public contract
- overlap 正确性依赖 DMA-local sequence state 和 destination-line hazard gate

当前 **不能** 假设的事实：

- 不能假设 staged `fill` 已支持
- 不能假设 staged `transpose` 已支持
- 不能假设 `CommandStage` 自动覆盖未来任意 staged DMA 扩展
- 不能假设当前 overlap 引入了新的 shared scheduler guarantees
- 不能假设公开了新的 explicit bank-selection ISA
