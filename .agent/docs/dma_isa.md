# DMA 模块宏指令 ISA / Live Reference (DMA 2.6, 512-bit)

## 1. 文档定位

本文档描述当前仓库 `src/npu/DmaUnit.*` 的 **live implementation-aligned**
DMA ISA / reference 事实。

它的用途是统一说明：

- `DmaUnit` 当前实际解析的宏指令格式
- `DmaUnit` 当前实际实现的 DMA 语义边界
- DMA testcase / workload 当前应生成的 staged 命令含义

当前文档层级：

- `.agent/plan/dma2.1_plan_v1.md` 仍保留为 DMA 2.1 的控制性计划文档
- `.agent/plan/dma2.1_plan_v1.md` **没有**同步更新 `CommandStage`、staged DMA、DMA-local overlap
- 本文档是 DMA 2.6 当前实现现实的 reference 文档
- 若计划文档与 live code 存在差异，以本文件和 `src/npu/DmaUnit.*` 为准

本文件不是开发过程记录，而是当前实现可依赖的 reference。

---

## 2. 当前 DMA 2.6 范围

当前 live DMA 仍保留 DMA 2.1 的基础边界：

- 所有 DMA 模式仍为 `INT8-only`
- 支持模式仍为：`move_layout`、`transpose`、`fill`
- `bank_size` 仍是 DMA-local workspace capacity 的权威控制参数
- 外部 `DRAM` / `SPM` fill 仍必须保持 `reads = 0`
- transpose 仍要求 `src_k == 0 && dst_k == 0`
- 不引入 shared `SEU` / `MegaCmdQueue` / launch plumbing 改造

在此基础上，当前 live DMA 额外实现了：

- `op_code[1:0] = CommandStage`
- staged `move_layout`
- staged `transpose`
- DMA-local、batch-level overlap

当前 **未实现**：

- staged `fill`
- line-level overlap
- streaming overlap
- shared scheduler / shared queue 语义改造
- 新的 public command-stage 编码以外的 staged ISA 扩展

---

## 3. 512-bit 宏指令布局

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

基础地址空间约束：

- DRAM: `0x2000_0000` ~ `0x5FFF_FFFF`
- SPM: `0x6000_0000` ~ `0x6FFF_FFFF`
- 外部访存以 64B cache line 为基本读写单位

---

## 4. DMA-specific `op_code[23:16]` 布局

当前 live `DmaUnit` 对 `op_code[23:16]` 的解释为：

- `op_code[7:5] = data_type`
- `op_code[4:2] = mode`
- `op_code[1:0] = CommandStage`

### 4.1 `data_type`

- `000 = INT8`
- 其他编码保留并必须被拒绝

### 4.2 `mode`

- `000 = move_layout`
- `001 = transpose`
- `010 = fill`
- `011..111 = reserved`

### 4.3 `CommandStage`

- `00 = Legacy`
- `01 = Load`
- `10 = Compute`
- `11 = Store`

---

## 5. CommandStage

### 5.1 CommandStage 是什么

`CommandStage` 是当前 live DMA 对 `op_code[1:0]` 的解释。它不是保留位；它是 DMA 当前真实使用的 staged 控制面。

### 5.2 CommandStage 的作用

`CommandStage` 的作用有两个：

- 选择该宏指令是 legacy 完整流水，还是 staged `Load` / `Compute` / `Store`
- 把宏指令分类到当前 SEU 模型下的不同 macro/issue queue 上，以便 DMA-local overlap 成立

当前 `DmaUnit` 的 queue 分类是：

- `Legacy -> Exec queue`
- `Load -> Load queue`
- `Compute -> Exec queue`
- `Store -> Store queue`

实现上对应 `classifyMacroCmd()` / `classifyIssueQueue()`。

### 5.3 CommandStage 的当前语义

#### `Legacy`

- 保持原有单条宏指令的完整 DMA pipeline 语义
- `move_layout`、`transpose`、`fill` 都仍支持 legacy 路径

#### `Load`

- staged `move_layout`：只执行当前下一 batch 的 source/destination line reads
- staged `transpose`：只执行当前下一 iteration 的 source/destination line reads
- staged `fill`：不支持

#### `Compute`

- staged `move_layout`：只执行当前下一 loaded batch 的 layout transform
- staged `transpose`：只执行当前下一 loaded iteration 的 transpose 计算
- staged `fill`：不支持

#### `Store`

- staged `move_layout`：只执行当前下一 computed batch 的 external writes
- staged `transpose`：只执行当前下一 computed iteration 的 external writes
- staged `fill`：不支持

### 5.4 当前为什么 2 bits 足够

对 **当前 DMA 2.6 scope**，2 bits 足够，因为 live 实现只需要四种编码：

- `Legacy`
- `Load`
- `Compute`
- `Store`

这四种值已经完整覆盖当前 staged 支持的两类 mode：

- staged `move_layout`
- staged `transpose`

也完整覆盖当前 overlap 需要的三段式流水：

- 读
- 算
- 写

这个“足够”结论只对当前 scope 成立。

不能据此推断：

- 未来任意 staged DMA 扩展都只需要 2 bits
- 未来若增加更细 phase、barrier/fence、mode-specific extra phases 时，当前编码仍然自动足够

若将来 staged DMA 需要额外 externally encoded phases，必须重新评估编码并重新冻结文档。

---

## 6. `mode_cfg` / `bank_cfg`

### 6.1 `move_layout`

`mode_cfg[5:0]`：

- `[0] src_mem_space`: `0 = DRAM`, `1 = SPM`
- `[1] dst_mem_space`: `0 = DRAM`, `1 = SPM`
- `[3:2] src_cut_dim`: `0 = H`, `1 = W`, `2 = C`, `3 = reserved`
- `[5:4] dst_cut_dim`: `0 = H`, `1 = W`, `2 = C`, `3 = reserved`
- `[31:6] reserved = 0`

`bank_cfg`：

- `bank_cfg` 必须为 `0`
- 这对 legacy `move_layout` 和 staged `move_layout` 都成立
- staged `move_layout` 的内部 batch slots / workspace 仍是 DMA-local 实现细节，不形成新的 public bank-selection 合约

### 6.2 `transpose`

`mode_cfg[5:0]`：

- `[0] src_mem_space`: `0 = DRAM`, `1 = SPM`
- `[1] dst_mem_space`: `0 = DRAM`, `1 = SPM`
- `[3:2] transpose_dim_a`: `0 = H`, `1 = W`, `2 = C`, `3 = reserved`
- `[5:4] transpose_dim_b`: `0 = H`, `1 = W`, `2 = C`, `3 = reserved`
- `[31:6] reserved = 0`

`bank_cfg[7:0]`：

- `[3:0] src_bank_id`
- `[7:4] dst_bank_id`
- `[31:8] reserved = 0`

当前 transpose 的共同约束：

- `src_bank_id < num_banks`
- `dst_bank_id < num_banks`
- `src_bank_id != dst_bank_id`
- `transpose_dim_a != transpose_dim_b`
- `src_k == 0 && dst_k == 0`
- `shape_h * shape_w * shape_c <= bank_size`

### 6.3 `fill`

`mode_cfg[1:0]`：

- `0 = DRAM`
- `1 = SPM`
- `2 = DMA_BANK`
- `3 = reserved`

`bank_cfg[3:0]`：

- `[3:0] dst_bank_id`
- `[31:4] reserved = 0`

当前 fill 的共同约束：

- 只支持 `Legacy`
- staged fill 必须被拒绝
- `src_base_addr == 0`
- source layout 字段必须为 `0`
- `dst_k == 0`
- 外部 fill 仍必须保持 `reads = 0`

---

## 7. 当前 staged 支持矩阵

| 模式 | Legacy | Staged Load/Compute/Store | Overlap | 当前边界 |
| :--- | :--- | :--- | :--- | :--- |
| `move_layout` | 支持 | 支持 | 支持 | DMA-local、batch-level only |
| `transpose` | 支持 | 支持 | 支持 | DMA-local、batch-level only |
| `fill` | 支持 | 不支持 | 不支持 | staged fill 直接拒绝 |

其他 `mode` 编码当前仍是 reserved，并在解析/校验阶段被拒绝。

---

## 8. staged `move_layout`

### 8.1 语义

staged `move_layout` 需要一组匹配的三条宏指令：

- `Load`
- `Compute`
- `Store`

当前实现要求：

- staged sequence 必须从 `Load` 开始
- 后续 `Compute` / `Store` 必须与 active sequence descriptor 完全匹配
- staged sequence 活跃期间不允许插入不属于该 sequence 的其他 DMA 命令
- 当前实现有效地只允许一个 active staged DMA sequence

### 8.2 运行时

- `Load` 读当前下一 batch 的 `sourceLines` 和 `destLines`
- `Compute` 在 DMA-local iteration plan 上执行 layout transform
- `Store` 写当前下一 computed batch 的 `destLines`

`destLines` 在 staged `Load` 中同样会被读取，因为目标 cache line 可能只被部分覆盖；未覆盖字节必须被保留。

### 8.3 overlap

当前 overlap 是：

- `Store(N)` 与 `Load(N+1)` 在不同 queue 上并行
- 仅限 batch-level
- 不扩展为 line-level / streaming overlap

---

## 9. staged `transpose`

### 9.1 语义

staged `transpose` 同样使用：

- `Load`
- `Compute`
- `Store`

约束与 staged `move_layout` 同类：

- staged sequence 必须从 `Load` 开始
- 后续 `Compute` / `Store` 必须与 active staged transpose descriptor 完全匹配
- staged sequence 活跃期间不允许 interleaving 的其他 DMA 命令

### 9.2 staged Load 的含义

staged transpose 的 `Load` **不是只读源 tensor**。

它必须读取：

- transpose 源 line
- transpose 目标 line

原因是 transpose 目的 cache line 可能被部分覆盖。若不先读目的 line，未覆盖字节无法保留。

这就是“staged Load includes all correctness-required reads”的当前 live 含义。

### 9.3 staged Compute / Store

- `Compute` 对当前已加载 iteration 执行 transpose 变换
- `Store` 写回当前已计算 iteration 的目标 lines

### 9.4 transpose-specific correctness rules

#### destination-line preservation

- staged transpose 会保留 legacy transpose 已有的 destination-line preservation 语义
- staged `Load` 读回 `destLines`
- staged `Compute` 只覆盖映射到 transpose 结果的字节
- staged `Store` 写回更新后的完整 64B line

#### destination-line hazard gating

当前实现存在 DMA-local `destLineHazard()` gating：

- 若 `Load(N+1)` 需要读取的某条 destination cache line
- 与当前 `Store(N)` 正在写回的 destination cache line 相同
- 则 `Load(N+1)` 必须等待，不允许制造不安全 overlap

这条规则是当前 transpose overlap 正确性的核心保护之一。

#### aliasing rule

当前 staged transpose 还额外施加了一个保守 aliasing 规则：

- 在 sequence 建立时，先构建全部 transpose iteration plans
- 若任意 `plan.sourceLines[].lineAddr`
- 与任意 `plan.destLines[].lineAddr`
- 在整个计划集合中发生相等
- 则该 staged transpose sequence 直接拒绝

当前报错文本为：

- `DmaUnit: staged transpose requires source/destination cache-line disjointness`

该规则只对 staged transpose 生效；它是当前 live 实现的保守边界，不应被误写成更宽松的语义。

### 9.5 overlap

当前 transpose overlap 是：

- DMA-local only
- batch-level only
- `Store(N)` 与 `Load(N+1)` 可能并行
- 若命中 destination-line hazard，则 overlap 必须被抑制

---

## 10. overlap 观测与证明边界

当前 DMA overlap 不依赖 debug log 作为唯一依据。

live `DmaUnit` 通过 Python 导出：

- `observedBatchOverlapCount()`

该计数器只在以下条件满足时增加：

- staged `Load` / `Store` 属于相邻 iteration 或 batch
- 两者都已经形成实际 in-flight memory transactions

`DMA_OVERLAP_OBSERVE ...` debug 行仍可出现，但它只是辅助观测，不是唯一正确性依据。

---

## 11. 当前必须保持的边界

未来开发者当前 **可以依赖**：

- `op_code[1:0]` 当前就是 `CommandStage`
- staged `move_layout` 已实现
- staged `transpose` 已实现
- staged `fill` 当前不支持
- overlap 当前仅为 DMA-local、batch-level
- `move_layout` 不引入新的 public bank-selection 合约
- staged transpose 当前要求 source/destination cache-line disjointness

未来开发者当前 **不能假设**：

- shared `SEU` / `MegaCmdQueue` 调度被改造过
- line-level overlap 已实现
- streaming overlap 已实现
- staged fill 已实现
- 2-bit `CommandStage` 对任意未来 staged DMA 扩展都天然足够

