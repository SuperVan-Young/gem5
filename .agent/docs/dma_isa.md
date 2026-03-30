# DMA 模块宏指令 ISA 参考 (DMA 2.1 v1, 512-bit)

## 1. 目的与控制关系

本文档是 DMA 2.1 的 ISA / reference 文档，用于给 `DmaUnit` 解析、校验、DMA-local workload、DMA-local test 提供统一的命令格式与语义参考。

控制关系：

- 当前阶段的 **控制性计划/规格文档** 是 `/.agent/plan/dma2.1_plan_v1.md`
- 本文档从该计划派生，用于做 ISA/reference 汇总
- 若本文与计划出现冲突，以 `dma2.1_plan_v1.md` 为准
- 实现执行顺序、blocker policy、共享边界约束仍以计划文档为主

DMA 2.1 v1 的目标语义：

- 保持 DMA 变更尽量 DMA-local
- 保留 `move_layout` / `transpose` / `fill` 三种模式
- 引入外部 `DRAM` / `SPM` / 内部 `DMA_BANK` fill 目标空间
- 去除 `buffer_size`，改用 bank-local workspace 语义
- 支持 generalized 3D transpose
- 保留 transpose 的 `k == 0` 约束
- 为 transpose 提供稳定的延迟参考模型

DMA 2.1 v1 明确不包含：

- partial multi-width DMA 语义
- `INT16` / `INT32` DMA 执行语义
- read-modify-write external fill
- transpose 与 blocked-layout 语义组合
- tensor descriptor
- 泛化 multiport 调度
- line-level / streaming overlap
- `SpecializedExecutionUnit` / `MegaCmdQueue` 重构

---

## 2. 基础架构假设

### 2.1 地址空间

- DRAM: `0x2000_0000` ~ `0x5FFF_FFFF`
- SPM: `0x6000_0000` ~ `0x6FFF_FFFF`
- DMA 宏指令中的外部地址字段仍按 32-bit 物理地址编码
- `DMA_BANK` 是 DMA internal workspace，不是外部物理地址空间
- `Word 1` / `Word 2` 仍表示外部 `DRAM` / `SPM` 地址字段
- 当 `fill` 目标是 `DMA_BANK` 时，`dst_base_addr` 必须为 `0`

### 2.2 字编号约定

当前 gem5 NPU 统一采用：

- `Word 0 = bits <31:0>`
- `Word 1 = bits <63:32>`
- ...
- `Word 15 = bits <511:480>`

DMA 2.1 v1 继续采用该约定。

### 2.3 通用访存与布局约束

- DMA 外部访存仍以 64B cache line 为基本读写单位
- `shape + stride + k` 仍是 layout 描述机制
- DMA 2.1 v1 当前 **所有模式均为 INT8-only**
- `data_type = 000` 是 DMA 2.1 v1 唯一合法编码
- `move_layout` 继续使用显式 cut-dimension 解释 `src_k` / `dst_k`
- `transpose` 扩展为 generalized 3D transpose，但仍要求 `src_k == 0` 且 `dst_k == 0`
- `fill` 对外部 `DRAM` / `SPM` 的实现必须保持 `reads = 0`
- 因此，外部 fill 仅在 fill region 完整覆盖所有触及的 64B cache line 时合法
- DMA 2.1 v1 不允许 external fill 的 read-before-write 或 read-modify-write
- DMA 2.1 的临时 workspace 模型改为 bank-local；`buffer_size` 不再是参考语义的一部分
- `bank_size` 是 DMA-local workspace capacity 的权威控制参数

---

## 3. 512-bit 宏指令布局

| Word | 字段 | 位段 | 含义 |
| :--- | :--- | :--- | :--- |
| `Word 0` | `Common_Header` | `<31:0>` | 公共头部；DMA-specific `op_code` 解释见下文 |
| `Word 1` | `src_base_addr` | `<63:32>` | 外部源地址；`fill` 中必须为 `0` |
| `Word 2` | `dst_base_addr` | `<95:64>` | 外部目的地址；`fill->DMA_BANK` 中必须为 `0` |
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
| `Word 15` | `fill_value_or_reserved` | `<511:480>` | `fill` 中低 8 位为 `fill_value_int8`；其余模式保留为 `0` |

---

## 4. Word 0: DMA-specific opcode 解释

`Word 0` 仍沿用公共头部格式：

- `device_type[31:28] = 0b0100` (DMA)
- `device_id[27:24]`
- `op_code[23:16]`
- `sync_indicator[15:8]`
- `set_indicator_sns[7]`
- `set_indicator_snd[6]`
- 其余保留

DMA 2.1 v1 对 `op_code[23:16]` 的解释：

- `op_code[7:5] = data_type`
- `op_code[4:2] = mode`
- `op_code[1:0] = reserved`

### 4.1 data_type

- `000 = INT8`
- 其他编码在 DMA 2.1 v1 中保留并必须被拒绝

### 4.2 mode

- `000 = move_layout`
- `001 = transpose`
- `010 = fill`
- `011..111 = reserved`

---

## 5. Word 13 = mode_cfg

`Word 13` 为 mode-specific 控制字。

### 5.1 move_layout

`mode = 000`

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

### 5.2 transpose

`mode = 001`

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

### 5.3 fill

`mode = 010`

- `[1:0] dst_mem_space`
  - `0 = DRAM`
  - `1 = SPM`
  - `2 = DMA_BANK`
  - `3 = reserved`
- `[31:2] reserved = 0`

---

## 6. Word 14 = bank_cfg

### 6.1 move_layout

`mode = 000`

- `bank_cfg` 必须为 `0`
- `move_layout` ISA 不暴露显式 bank 选择字段

### 6.2 transpose

`mode = 001`

- `[3:0] src_bank_id`
- `[7:4] dst_bank_id`
- `[31:8] reserved = 0`

规则：

- `src_bank_id < num_banks`
- `dst_bank_id < num_banks`
- `src_bank_id != dst_bank_id`

### 6.3 fill

`mode = 010`

- `[3:0] dst_bank_id`
- `[31:4] reserved = 0`

规则：

- 若 `dst_mem_space = DMA_BANK`，则 `dst_bank_id < num_banks`
- 若 `dst_mem_space = DRAM` 或 `SPM`，则 `dst_bank_id == 0`

---

## 7. Word 15 = fill_value / reserved

### 7.1 move_layout / transpose

- `Word 15 == 0`
- 不得消费 `Word 15`

### 7.2 fill

- `Word 15[7:0] = fill_value_int8`
- `Word 15[31:8] = 0`
- `fill=0` 可直接表示为 `Word 15 = 0`

DMA 2.1 v1 不定义 `INT16` / `INT32` fill-value 解释。

---

## 8. Per-mode 参考语义

## 8.1 move_layout

### 外部端点

- 仅允许外部 `DRAM` / `SPM` 端点
- `src_mem_space` / `dst_mem_space` 必须与 `src_base_addr` / `dst_base_addr` 的实际地址范围一致

### layout 语义

- `shape + stride + k` 仍是 layout 描述机制
- `src_k` 按 `src_cut_dim` 解释
- `dst_k` 按 `dst_cut_dim` 解释
- blocked-layout 合法性必须针对选中的 cut-dimension 显式检查
- 非 canonical blocked-layout 不允许静默猜测解释

### workspace 语义

- `move_layout` 仍不暴露 ISA 级显式 bank 选择
- DMA 2.1 参考模型中，其 DMA-local workspace 已从 flat temporary buffer 语义迁移为 bank-local workspace 语义
- `bank_size` 是该 DMA-local workspace 的 capacity 控制参数
- `buffer_size` 不再属于 DMA 2.1 参考语义

### 校验要求

- `src_cut_dim` / `dst_cut_dim` 必须在 `0..2`
- 所有保留位必须为 `0`
- 地址范围必须与声明的 memory space 匹配

## 8.2 transpose

### 外部端点

- 外部输入来自 `DRAM` / `SPM`
- 外部输出写回 `DRAM` / `SPM`
- `src_mem_space` / `dst_mem_space` 必须与地址范围一致

### generalized 3D 语义

- 选择两个维度 `dim_a` 与 `dim_b`
- 第三个维度 `dim_rest` 作为 outer loop
- 对每个 `dim_rest` slice 执行一次 transpose kernel
- 例子：
  - `H <-> W` 时对 `C` 做 outer loop
  - `H <-> C` 时对 `W` 做 outer loop
  - `W <-> C` 时对 `H` 做 outer loop

### bank 语义

- 源张量先暂存到 `src_bank_id`
- transpose 结果写入 `dst_bank_id`
- 最终结果从 `dst_bank_id` 写回外部 `dst_base_addr`
- bank workspace 从内部 offset `0` 开始
- bank 内容不形成软件可见的持久状态

### 约束

- `transpose_dim_a != transpose_dim_b`
- `transpose_dim_a` / `transpose_dim_b` 必须在 `0..2`
- `src_k == 0`
- `dst_k == 0`
- `required_bytes <= bank_size`

## 8.3 fill

### 通用语义

- `fill` 继续使用 `shape + stride` 描述目标 region
- source fields 在 `fill` 中无效并必须为 `0`
- `fill` 仅使用 INT8 fill value 语义

### DMA_BANK fill

- 目标空间为 `DMA_BANK`
- `dst_base_addr == 0`
- `dst_bank_id < num_banks`
- region 从 bank 内部 offset `0` 开始
- `required_bytes <= bank_size`
- `DMA_BANK` 内容仍是 command-local only
- 本文档不把 `DMA_BANK` 定义为软件可见持久状态

### 外部 DRAM / SPM fill

- `dst_mem_space` 必须与 `dst_base_addr` 的地址范围一致
- `dst_bank_id == 0`
- 这是实际外部写操作：
  - `reads = 0`
  - `writes > 0`
- DMA 2.1 v1 仅允许 full-cache-line external fill
- 若 fill region 仅覆盖某条 64B cache line 的一部分，则该命令必须被拒绝
- DMA 2.1 v1 不允许 external fill 的 read-before-write / read-modify-write

### fill 校验要求

- `mode_cfg` 保留位必须为 `0`
- `bank_cfg` 保留位必须为 `0`
- `Word 15[31:8]` 必须为 `0`
- `Word 1 = 0`
- `Word 6..8 = 0`
- `Word 12 = 0`
- 目标空间与地址/`dst_bank_id` 规则必须一致

---

## 9. Transpose 延迟参考

DMA 2.1 参考延迟模型：

```text
transpose_total_latency =
transpose_unit_latency *
extent(dim_a) *
extent(dim_b) *
extent(dim_rest)
```

其中：

- `transpose_unit_latency` 为可配置参数
- `extent(dim_x)` 从 command shape 读取
- 该公式保留 DMA 2.0 2D 情况的自然扩展形式

若实现提供稳定可观测的延迟输出，推荐包含：

- `dim_a`
- `dim_b`
- `extent_a`
- `extent_b`
- `extent_rest`
- `transpose_unit_latency`
- `computed_total_latency`

---

## 10. 参数参考模型

DMA 2.1 参考参数：

- `num_banks`
- `bank_size`
- `transpose_unit_latency`

说明：

- `buffer_size` 已从 DMA 2.1 参考模型中移除
- bank-local workspace semantics 由 `num_banks` 与 `bank_size` 定义
- 具体实现阶段的执行纪律、blocker rule、共享边界限制，仍以 `dma2.1_plan_v1.md` 为准
