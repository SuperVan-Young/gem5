# DMA 模块宏指令 ISA (512-bit)

## 1. 架构假设与设计前提

本 ISA 设计基于以下底层存储与总线架构的物理约定：

1. **物理地址空间划分 (32-bit)**:
* **DRAM (主存)**: 1 GB 容量，基地址挂载于 `0x2000_0000`，范围 `0x2000_0000` ~ `0x5FFF_FFFF`。
* **SPM (Scratchpad Memory)**: 256 MB 容量，基地址挂载于 `0x6000_0000`，范围 `0x6000_0000` ~ `0x6FFF_FFFF`。


2. **地址映射机制**:
* 采用虚拟地址到物理地址的**恒等映射 (Identity Mapping)**，即 VA = PA。
* 基于以上两点，系统最大有效物理地址不会超过 `0x6FFF_FFFF`，因此**地址字段被安全压缩至 32-bit**，从而释放大量总线指令空间。


3. **数据位宽约束**:
* 底层 DMA 访存位宽按 **64 Bytes** 对齐，需结合 `data_type` 计算 Burst 传输的元素个数。



---

## 2. 字编号约定与宏指令位段分配表 (Total: 512 bits)

### 2.1 当前主线约定

当前 gem5 NPU 主线统一使用如下 `Word` 编号方式：

- `Word 0 = bits <31:0>`
- `Word 1 = bits <63:32>`
- ...
- `Word 15 = bits <511:480>`

DMA 早期文档曾使用相反的命名方式，即把同一条 512-bit 指令中的最高 32-bit lane 记作 `Word 15`，最低 32-bit lane 记作 `Word 0`。那是**字编号/命名约定差异**，不是 512-bit 宏指令物理布局发生了变化。

本文件现在统一切换到当前主线约定。对应关系如下：

- 旧文档 `Word 15 (Common_Header)` = 当前 `Word 0`
- 旧文档 `Word 14 (src_base_addr)` = 当前 `Word 1`
- ...
- 旧文档 `Word 3 (dma_block_cfg)` = 当前 `Word 12`

也就是说，本次文档修订的目标是**统一命名与解析索引**，而不是重新设计 DMA payload 语义。

### 2.2 宏指令位段分配表

为了方便硬件解码器 (SystemVerilog) 截取和软件端 (gem5 C++ 模型 / 测试侧命令打包) 解析，除公共头部外，DMA 专属控制字段全部严格对齐到 **32-bit (Word) 边界**。

| 当前字编号 | 字段名称 (Field) | 位偏移 (Bits) | 位宽 | 字段含义与硬件动作解释 |
| :--- | :--- | :--- | :--- | :--- |
| **Word 0** | `Common_Header` | `<31:0>` | 32 | **公共头部**：<br>• `device_type`<31:28> = `4'b0100` (DMA)<br>• `device_id`<27:24>: 硬件实例 ID<br>• `op_code`<23:16> 被 DMA 重定义为 3 部分：<br>  - `[7:5] data_type`: `000`(INT8)，其他保留<br>  - `[4:2] xfer_mode`: `000`(D->S)，`001`(S->D)，`010`(S->S)，`011`(D->D)，其他保留用于未来拓展<br>  - `[1:0]`: 保留位<br>• `sync_indicator`<15:8>: 同步信号量编号<br>• `set_indicator_sns`<7>: 指令执行完成后置位同步表项（与当前主线 sync bit 语义保持一致）<br>• `set_indicator_snd`<6>: 指令执行完成后置位同步表项（与当前主线 sync bit 语义保持一致）<br>• `<5:0>`: 保留字段 |
| **Word 1** | `src_base_addr` | `<63:32>` | 32 | **源起始物理地址**。支持指针直传。<br>*(范围校验: `0x20000000`~`0x6FFFFFFF`)* |
| **Word 2** | `dst_base_addr` | `<95:64>` | 32 | **目的起始物理地址**。 |
| **Word 3** | `tensor_shape_h` | `<127:96>` | 32 | 三维张量逻辑高度 (**H 轴长度**)。 |
| **Word 4** | `tensor_shape_w` | `<159:128>` | 32 | 三维张量逻辑宽度 (**W 轴长度**)。 |
| **Word 5** | `tensor_shape_c` | `<191:160>` | 32 | 三维张量逻辑通道数 (**C 轴长度**)。 |
| **Word 6** | `src_stride_h` | `<223:192>` | 32 | 源数据在 **H** 轴的物理跨度 (Stride)。 |
| **Word 7** | `src_stride_w` | `<255:224>` | 32 | 源数据在 **W** 轴的物理跨度 (Stride)。 |
| **Word 8** | `src_stride_c` | `<287:256>` | 32 | 源数据在 **C** 轴的物理跨度 (Stride)。 |
| **Word 9** | `dst_stride_h` | `<319:288>` | 32 | 目的数据在 **H** 轴的物理跨度 (Stride)。 |
| **Word 10** | `dst_stride_w` | `<351:320>` | 32 | 目的数据在 **W** 轴的物理跨度 (Stride)。 |
| **Word 11** | `dst_stride_c` | `<383:352>` | 32 | 目的数据在 **C** 轴的物理跨度 (Stride)。 |
| **Word 12** | `dma_block_cfg` | `<415:384>` | 32 | **DMA 块维度配置字**：<br>• `[31:16] dst_k`: 目的端 Block 维度 (16 bits)。`0` 代表标准 HWC 格式；`>0` 代表 Blocked 格式，值为具体的 $k$。<br>• `[15:0] src_k`: 源端 Block 维度 (16 bits)。`0` 代表标准 HWC 格式；`>0` 代表 Blocked 格式，值为具体的 $k$。 |
| **Word 13~15** | `Reserved` | `<511:416>` | 96 | **保留字段**。拥有 96 bits 的连续保留空间。 |

---

## 3. 核心设计说明

### 3.1 对齐与无截断解析

由于释放了 64 位地址带来的压力，将 `Shape` 和 `Stride` 统一扩展至 32-bit，并且对齐到 Word 边界。

### 3.2 Layout Transformation

ISA 强制要求分离 `src_stride` 和 `dst_stride`。

### 3.3 Data Type

加入 `data_type` 控制位。由于硬件具有 64 Bytes 的物理位宽限制，DMA AXI 接口在攒够一个 Cache Line 前需要精准知道当前元素的字节数（如 INT8 需 64 个元素打一拍，INT32 仅需 16 个元素打一拍）。
