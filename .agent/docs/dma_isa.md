# DMA 模块宏指令 ISA (DMA 2.0 v1, 512-bit)

## 1. 目的与范围

本文冻结 DMA 2.0 v1 的宏指令语义，作为后续 `DmaUnit` 解析、校验、DMA-local workload、DMA-local test 的统一语义来源。

DMA 2.0 v1 的目标：

- 保持当前主线 SEU 队列与完成同步语义不变
- 在 DMA 范围内引入 `move_layout` / `transpose` / `fill`
- 引入显式 cut-dimension 控制
- 引入 banked internal workspace
- 保持 `Word 1..12` 与当前 DMA 的 shape / stride / `k` 用法尽量兼容

DMA 2.0 v1 明确不包含：

- 泛化 multiport 调度
- tensor descriptor
- `SpecializedExecutionUnit` / `MegaCmdQueue` 重构
- token-order coupling 修复
- 共享测试基础设施修复
- `Word 15` 的实际消费
- 非零 fill value
- 持久、软件可见的 DMA bank 存储语义

---

## 2. 基础架构假设

### 2.1 地址空间

- DRAM: `0x2000_0000` ~ `0x5FFF_FFFF`
- SPM: `0x6000_0000` ~ `0x6FFF_FFFF`
- 当前 DMA 宏指令中的地址字段仍按 32-bit 物理地址编码
- `Word 1` / `Word 2` 始终表示**外部** DRAM/SPM 地址，不表示 DMA bank 内部地址

### 2.2 字编号约定

当前 gem5 NPU 主线统一采用：

- `Word 0 = bits <31:0>`
- `Word 1 = bits <63:32>`
- ...
- `Word 15 = bits <511:480>`

DMA 2.0 v1 继续采用该约定。

### 2.3 访存与布局约束

- DMA 外部访存仍按 64B cache line 对齐/收发
- `shape + stride + k` 仍是唯一的 layout 描述机制
- DMA 2.0 v1 中，`move_layout` 通过显式 cut-dimension 决定 `src_k` / `dst_k` 的解释维度
- DMA 2.0 v1 中，`transpose` v1 不与 blocked-layout 语义组合
- DMA 2.0 v1 中，`fill` 仅作用于 DMA 内部 bank workspace

---

## 3. 512-bit 宏指令布局

| Word | 字段 | 位段 | 含义 |
| :--- | :--- | :--- | :--- |
| `Word 0` | `Common_Header` | `<31:0>` | 公共头部；DMA-specific `op_code` 解释见下文 |
| `Word 1` | `src_base_addr` | `<63:32>` | 外部源地址 |
| `Word 2` | `dst_base_addr` | `<95:64>` | 外部目的地址 |
| `Word 3` | `tensor_shape_h` | `<127:96>` | 逻辑 H |
| `Word 4` | `tensor_shape_w` | `<159:128>` | 逻辑 W |
| `Word 5` | `tensor_shape_c` | `<191:160>` | 逻辑 C |
| `Word 6` | `src_stride_h` | `<223:192>` | 源 H stride |
| `Word 7` | `src_stride_w` | `<255:224>` | 源 W stride |
| `Word 8` | `src_stride_c` | `<287:256>` | 源 C stride |
| `Word 9` | `dst_stride_h` | `<319:288>` | 目的 H stride |
| `Word 10` | `dst_stride_w` | `<351:320>` | 目的 W stride |
| `Word 11` | `dst_stride_c` | `<383:352>` | 目的 C stride |
| `Word 12` | `dma_block_cfg` | `<415:384>` | `[31:16] dst_k`, `[15:0] src_k` |
| `Word 13` | `mode_cfg` | `<447:416>` | DMA 2.0 模式控制字 |
| `Word 14` | `bank_cfg` | `<479:448>` | DMA 2.0 bank 控制字 |
| `Word 15` | `Reserved` | `<511:480>` | DMA 2.0 v1 中必须为 `0` |

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

DMA 2.0 v1 对 `op_code[23:16]` 的解释如下：

- `op_code[7:5] = data_type`
- `op_code[4:2] = mode`
- `op_code[1:0] = reserved`

### 4.1 data_type

- `000 = INT8`
- 其他编码在 DMA 2.0 v1 中保留并应被拒绝

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

位段定义：

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

位段定义：

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

DMA 2.0 v1 的 fill 保持最小契约：

- `[31:0] reserved = 0`

DMA 2.0 v1 不在 `mode_cfg` 中为 fill 定义 cut-dimension 或 fill-value 字段。

---

## 6. Word 14 = bank_cfg

### 6.1 move_layout

`mode = 000`

- `bank_cfg` 必须为 `0`
- `move_layout` 不暴露显式 bank 选择

### 6.2 transpose

`mode = 001`

位段定义：

- `[3:0] src_bank_id`
- `[7:4] dst_bank_id`
- `[31:8] reserved = 0`

### 6.3 fill

`mode = 010`

位段定义：

- `[3:0] dst_bank_id`
- `[31:4] reserved = 0`

---

## 7. Word 15

`Word 15` 在 DMA 2.0 v1 中保留。

要求：

- `Word 15 == 0`
- DMA 2.0 v1 任何实现都不得消费 `Word 15`
- 若后续确需使用，必须先修订计划并重新冻结 ISA

---

## 8. Per-mode 语义冻结

## 8.1 move_layout

### 外部端点

- 仅允许外部 DRAM / SPM 端点
- `src_mem_space` / `dst_mem_space` 必须与 `src_base_addr` / `dst_base_addr` 的实际地址范围一致

### layout 语义

- `shape + stride + k` 仍是唯一 layout 描述机制
- `src_k` 按 `src_cut_dim` 解释
- `dst_k` 按 `dst_cut_dim` 解释
- blocked-layout 合法性必须针对选中的 cut dimension 显式检查
- 非 canonical blocked-layout 不允许静默猜测解释

### bank 语义

- 不暴露显式 bank 使用
- `Word 14` 必须为 `0`

### 校验要求

- `src_mem_space` / `dst_mem_space` 取值合法
- `src_cut_dim` / `dst_cut_dim` 取值必须在 `0..2`
- 所有保留位必须为 `0`
- 地址范围与声明 memory space 必须匹配

## 8.2 transpose

### 外部端点

- 外部输入仍来自 DRAM/SPM
- 外部输出仍写回 DRAM/SPM
- `src_mem_space` / `dst_mem_space` 必须与地址范围一致

### 内部 bank 语义

- DMA internal banks 仅是 temporary workspace
- 源张量先暂存到 `src_bank_id`
- 内部 transpose 结果写入 `dst_bank_id`
- 最终结果从 `dst_bank_id` 写回外部 `dst_base_addr`
- bank workspace 从内部 offset `0` 开始
- bank 内容不形成软件可见的持久状态

### bank 约束

- `src_bank_id < num_banks`
- `dst_bank_id < num_banks`
- `src_bank_id != dst_bank_id`

### transpose 维度约束

- DMA 2.0 v1 只实现 2D transpose
- `transpose_dim_a != transpose_dim_b`
- `transpose_dim_a` / `transpose_dim_b` 必须在 `0..2`

### v1 实现约束

为了控制实现范围，DMA 2.0 v1 可以把实际执行与测试限制在：

- 非转置轴的 extent 为 `1`

这是一条 **v1 implementation constraint**，不是永久 ISA 限制；实现时必须明确记录。

### k 约束

DMA 2.0 v1 transpose 不与 blocked-layout 组合：

- `src_k == 0`
- `dst_k == 0`

### 容量约束

- 计算 `required_bytes`
- 必须满足 `required_bytes <= bank_size`

### 延迟模型

使用：

```text
transpose_latency = transpose_unit_latency * (extent(dim_a) * extent(dim_b))
```

其中：

- `transpose_unit_latency` 为可配置参数
- `extent(dim_x)` 从 command shape 读取

## 8.3 fill

### 作用范围

- DMA 2.0 v1 fill 仅写 DMA internal buffer
- 默认 fill value 为 `0`

### 内部 bank 语义

- 由 `dst_bank_id` 选择目标 bank
- workspace 从内部 offset `0` 开始
- bank 内容是 **command-local only**
- fill 不定义持久 bank 存储语义

### 外部访存语义

- 不发起外部读请求
- 不发起外部写请求

### v1 fill contract

DMA 2.0 v1 fill 采用窄契约，不做启发式解释：

- `src_base_addr == 0`
- `dst_base_addr == 0`
- source layout fields 未使用，必须为 `0`
- destination shape / stride fields 描述内部 fill region
- 所有保留位必须为 `0`

### 校验要求

- `dst_bank_id < num_banks`
- `required_bytes <= bank_size`
- 外部地址字段遵守上述 fill contract

---

## 9. SimObject / 参数冻结

DMA 2.0 v1 需要的 DMA 参数：

- `num_banks`
- `bank_size`
- `transpose_unit_latency`

迁移兼容建议：

- 可以暂时保留 `buffer_size` 作为 compatibility alias
- 若保留，必须明确其为过渡参数
- DMA 2.0 语义上，`bank_size` 应成为权威参数
- 不允许长期维持两套平行含义

---

## 10. Phase 0 冻结结论

后续实现阶段不得再对以下事项进行语义猜测：

- `Word 13` 的字段定义
- `Word 14` 的字段定义
- `Word 15` 是否可用
- `move_layout` 的 memory-space bits 与 cut-dim bits
- `transpose` 的 bank / dim / latency 约束
- `fill` 的窄契约与 command-local 语义

如需修改上述语义，必须先更新计划并重新冻结本文件。
