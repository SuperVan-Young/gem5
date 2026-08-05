# FlashAttention T7/F7 全流程性能与 PPA 评估报告

## 1. 执行摘要

本报告评估 gem5 NPU 上 FlashAttentionV2 在 T7 与 F7 工艺下的性能、
功耗、面积和能效，并进一步分析以下三档架构：

1. 32-bank baseline；
2. 64-bank double buffer；
3. 96-bank Vector-2x double buffer。

本轮只修改并重跑 PPA 模型，性能模型恢复并保持在上一版本。主要 PPA
修正如下：

- double buffer 使用两套完整的 32-bank SRAM，而非只增加少量容量；
- SRAM macro 的 floorplan 利用率设为 50%，实际占地为 macro 裸面积的
  2 倍；
- gem5 性能参数、SPM 带宽和流水模型维持上一版本不变；
- T7/F7 在同档架构和 2 GHz 频率下采用相同的性能结果，工艺差异仅进入
  PPA 计算。

核心结论：

- F7 相对同档 T7，功耗下降 45.98%–47.89%，面积下降
  38.27%–42.00%；
- 32-bank baseline 的实测性能为 35,440 cycles、7.574 TOPS；
- 64-bank double buffer 的流水模型为 26,544 cycles、10.113 TOPS，
  相对 baseline 提升 1.335 倍；
- 96-bank Vector-2x double buffer 的流水模型为 17,288 cycles、
  15.527 TOPS，相对 baseline 提升 2.050 倍；
- F7 Vector-2x double buffer 相对 T7 baseline，在功耗略低的情况
  下，性能提升 2.050 倍，延迟下降 51.22%，任务能耗下降 51.66%，
  面积仅增加 5.05%。

最后一项是最重要的系统结论：F7 本身不减少 cycle，但它释放出的功耗
和面积预算足以容纳 double buffer 与 Vector 扩展，从而在接近 T7
baseline 功耗的条件下获得约 2.05 倍的架构性能。

## 2. 评估对象与工作负载

### 2.1 FlashAttention 工作负载

默认评估任务为：

| 参数 | 数值 |
|---|---:|
| Batch | 1 |
| Attention heads | 1 |
| Query length | 256 |
| KV length | 1024 |
| Head dimension | 128 |
| Query tile（BR） | 128 |
| KV tile（BC） | 128 |
| Q/K/V 输入 | INT8 |
| Tensor 累加 | INT32 |
| Softmax | FP32 |
| 总 Tensor operations | 134,217,728 |

QK、online softmax 和 PV 按 `BR=BC=128` 分块。物理 MPU 为 64×64，
峰值为 4096 MAC/cycle。乘加按两个 operations 计，在 2 GHz 下的物理
峰值为 16.384 TOPS。

### 2.2 三档硬件配置

| 配置 | SRAM banks | 聚合带宽 | Buffer slots | Vector 能力 |
|---|---:|---:|---:|---:|
| Baseline | 32 | 128 B/cycle | 1 | 128 FP32 elements/cycle |
| Double buffer | 64 | 256 B/cycle | 2 | 128 FP32 elements/cycle |
| Vector-2x DB | 96 | 384 B/cycle | 2 | 256 FP32 elements/cycle |

Vector-2x 的 96 banks 分为：

- Tensor engine：32 banks，128 B/cycle；
- 两组 Vector datapath：64 banks，256 B/cycle。

## 3. 本轮模型修改

### 3.1 SRAM floorplan 利用率

PPA 配置新增：

```yaml
memory:
  sram:
    layout_utilization_pct: 50
```

模型现在同时输出：

- `macro_area`：SRAM macro 数据表中的裸面积；
- `floorplan_area`：考虑走线、间距和版图留白后的实际占地；
- 总芯片面积使用 `floorplan_area`。

计算方式为：

```text
floorplan_area = macro_area / 0.50
```

该系数只影响面积，不影响 SRAM 功耗。

### 3.2 性能模型保持上一版本

本轮不修改 gem5 的 SPM 带宽、执行单元参数或流水计算方法。SPM
继续使用上一版本配置：

```yaml
simulation:
  spm:
    bandwidth: 100GiB/s
```

32/64/96-bank 信息继续用于 buffer context 和 analytical overlap 判断；
PPA 则分别按 32/64/64 个物理 macro 计费。本轮不将 4 B/cycle/bank 重新映射为 gem5
`ScratchpadMemory.bandwidth`。这样可以确保性能结果与上一版报告可直接
比较，只有 PPA 面积口径发生变化。

### 3.3 Double buffer 容量

SRAM 数量由 bank 组织决定：

| 方案 | SRAM macros | 物理 macro 容量 |
|---|---:|---:|
| Baseline | 32 | 2 MiB |
| Double buffer | 64 | 4 MiB |
| Vector-2x DB | 64 | 4 MiB |

虽然 Vector-2x DB 有 96 个逻辑 bank，但它们由 64 个物理 macro 切分
得到，不再把每个逻辑 bank 等同于一个完整 macro。因此它与 double
buffer 都保持 4 MiB 物理容量；Vector 算力翻倍只增加计算与端口组织，
不再额外增加 SRAM 容量。Double buffer 的 SRAM macro 数量、裸面积、
floorplan 面积和 SRAM 功耗仍严格为 baseline 的两倍。

## 4. 实验方法及结果边界

### 4.1 gem5 实测部分

每组配置均实际运行完整 FlashAttentionV2 workload，并记录：

- QK busy cycles；
- online softmax busy cycles；
- PV busy cycles；
- operator busy cycles；
- NPU profile 与性能摘要。

F7 的 baseline、double buffer 和 Vector-2x double buffer 三组实验均
已重新运行并 PASS。T7 配置只用于 PPA；同档 T7/F7 在 2 GHz 下采用
相同的性能结果，因为当前性能模型不包含工艺相关的频率或延迟缩放。

### 4.2 分析流水部分

当前 workload 尚未真正并发发射 Tensor 与 Vector 阶段。因此报告严格
区分两类数据：

- **实测周期**：gem5 当前顺序 workload 实际运行的 busy cycles；
- **流水投影周期**：使用实测 QK、softmax、PV 阶段周期，按双 buffer
  有限流水公式计算的架构周期。

Double buffer 使用两组 BR-addressed context，流水为：

```text
QK0 -> max(QK1, Softmax0) -> max(Softmax1, PV0) -> PV1
```

Vector-2x 还将实测 softmax busy cycles 按两组 Vector datapath 分摊，
然后应用相同的双 buffer 流水公式。

因此，本报告中的 10.113 TOPS 和 15.527 TOPS 是“实测阶段周期 + 架构
流水模型”的投影结果，不应描述为当前 workload 已真实并发执行达到的
端到端吞吐。

## 5. gem5 性能实验结果

### 5.1 实测阶段周期

| 方案 | QK | Softmax | PV | 实际 operator busy |
|---|---:|---:|---:|---:|
| Baseline | 8,896 | 17,648 | 8,896 | 35,440 |
| Double buffer | 8,896 | 17,648 | 8,896 | 35,440 |
| Vector-2x DB | 8,608 | 17,360 | 8,608 | 34,576 |

Double buffer 当前仍顺序执行，因此其实测 operator busy cycles 与
baseline 相同。Vector-2x 配置增加了 SPM ports 和带宽，所以 QK/PV 的
实测周期明显下降；完整的 Vector 2 倍计算效果仍通过
`vector_compute_scale=2` 在分析模型中表达。

### 5.2 流水投影性能

| 方案 | 流水周期 | 延迟 @ 2 GHz | 吞吐 | 峰值利用率 | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| Baseline | 35,440 | 17.720 μs | 7.574 TOPS | 46.23% | 1.000× |
| Double buffer | 26,544 | 13.272 μs | 10.113 TOPS | 61.72% | 1.335× |
| Vector-2x DB | 17,288 | 8.644 μs | 15.527 TOPS | 94.77% | 2.050× |

Double buffer 的四个流水 stage 为：

```text
4,448 -> 8,824 -> 8,824 -> 4,448 cycles
```

总计 26,544 cycles，相比 baseline 节省 8,896 cycles，延迟下降
25.10%。

Vector-2x double buffer 的四个流水 stage 为：

```text
4,304 -> 4,340 -> 4,340 -> 4,304 cycles
```

总计 17,288 cycles，相比 baseline 延迟下降 51.22%。

### 5.3 与上一版性能结果的一致性

| 指标 | 上一版 | 本轮恢复后 | 变化 |
|---|---:|---:|---:|
| Baseline cycles | 35,440 | 35,440 | 0% |
| Double-buffer modeled cycles | 26,544 | 26,544 | 0% |
| Vector-2x modeled cycles | 17,288 | 17,288 | 0% |

本轮性能模型没有变化。所有新的面积、面积效率和能耗结论都在上一版
性能结果上重新计算。

## 6. T7/F7 PPA 结果

### 6.1 Baseline

| 指标 | T7 | F7 | F7 相对 T7 |
|---|---:|---:|---:|
| Power | 23.064 W | 12.459 W | -45.98% |
| Compute area | 2.729 mm² | 1.797 mm² | -34.14% |
| SRAM macro area | 0.670 mm² | 0.335 mm² | -50.00% |
| SRAM floorplan area | 1.339 mm² | 0.670 mm² | -50.00% |
| Total floorplan area | 4.068 mm² | 2.467 mm² | -39.36% |

### 6.2 Double buffer

| 指标 | T7 | F7 | F7 相对 T7 |
|---|---:|---:|---:|
| Power | 24.395 W | 13.124 W | -46.20% |
| SRAM floorplan area | 2.678 mm² | 1.339 mm² | -50.00% |
| Total floorplan area | 5.407 mm² | 3.136 mm² | -42.00% |

### 6.3 Vector-2x double buffer

| 指标 | T7 | F7 | F7 相对 T7 |
|---|---:|---:|---:|
| Power | 43.861 W | 22.857 W | -47.89% |
| SRAM floorplan area | 2.678 mm² | 1.339 mm² | -50.00% |
| Total floorplan area | 6.923 mm² | 4.273 mm² | -38.27% |

### 6.4 同档 T7/F7 效率对比

由于同档 T7 与 F7 的 cycle 完全相同，性能相同，Perf/W 和
Perf/mm² 的提升仅由功耗和面积下降决定：

| 方案 | F7 Perf/W 提升 | F7 Perf/mm² 提升 |
|---|---:|---:|
| Baseline | 1.851× | 1.649× |
| Double buffer | 1.859× | 1.724× |
| Vector-2x DB | 1.919× | 1.620× |

## 7. 全流程延迟、功耗、面积与能耗

下表中的延迟采用流水投影周期。任务能耗按：

```text
Energy = PPA reported power × projected latency
```

计算。因此它适合架构方案比较，但不是 post-layout workload power
simulation。

| 方案 | 制程 | Power | Floorplan area | 延迟 | 吞吐 | 能量/task |
|---|---|---:|---:|---:|---:|---:|
| Baseline | T7 | 23.064 W | 4.068 mm² | 17.720 μs | 7.574 TOPS | 408.7 μJ |
| Baseline | F7 | 12.459 W | 2.467 mm² | 17.720 μs | 7.574 TOPS | 220.8 μJ |
| Double buffer | T7 | 24.395 W | 5.407 mm² | 13.272 μs | 10.113 TOPS | 323.8 μJ |
| Double buffer | F7 | 13.124 W | 3.136 mm² | 13.272 μs | 10.113 TOPS | 174.2 μJ |
| Vector-2x DB | T7 | 43.861 W | 6.923 mm² | 8.644 μs | 15.527 TOPS | 379.1 μJ |
| Vector-2x DB | F7 | 22.857 W | 4.273 mm² | 8.644 μs | 15.527 TOPS | 197.6 μJ |

## 8. Incremental 修改与每步收益

### 8.1 第一步：T7 baseline → F7 baseline

这一阶段只更换工艺，不改变架构和周期性能：

| 指标 | 改善 |
|---|---:|
| 性能 | 0% |
| 延迟 | 0% |
| 功耗 | -45.98% |
| 面积 | -39.36% |
| 能耗 | -45.98% |
| Perf/W | 1.851× |
| Perf/mm² | 1.649× |

F7 的第一层价值是释放 PPA 预算，而不是直接减少 FlashAttention
cycles。

### 8.2 第二步：F7 baseline → F7 double buffer

| 指标 | 变化 |
|---|---:|
| 性能 | +33.51% |
| 延迟 | -25.10% |
| 功耗 | +5.34% |
| 总面积 | +27.14% |
| SRAM floorplan area | +100.00% |
| 能耗 | -21.10% |
| Perf/W | 1.267× |
| Perf/mm² | 1.050× |

这是最稳健的 incremental 架构升级：SRAM 确实翻倍，但功耗增加很小，
同时性能、能耗和面积效率均得到改善。

### 8.3 第三步：F7 double buffer → F7 Vector-2x DB

| 指标 | 变化 |
|---|---:|
| 性能 | +53.54% |
| 延迟 | -34.87% |
| 功耗 | +74.16% |
| 总面积 | +36.25% |
| 能耗 | +13.43% |
| Perf/W | 0.882× |
| Perf/mm² | 1.127× |

这一步以额外的计算资源换取绝对性能；相对 double buffer，能耗和
Perf/W 变差，但由于不再增加 SRAM 容量，Perf/mm² 反而提升 12.7%。

### 8.4 F7 baseline → F7 Vector-2x DB

| 指标 | 变化 |
|---|---:|
| 性能 | 2.050× |
| 延迟 | -51.22% |
| 功耗 | +83.46% |
| 面积 | +73.24% |
| 能耗 | -10.51% |
| Perf/W | 1.117× |
| Perf/mm² | 1.183× |

## 9. 最终跨工艺、跨架构结论

比较 T7 baseline 与 F7 Vector-2x double buffer：

| 指标 | T7 baseline | F7 Vector-2x DB | 总变化 |
|---|---:|---:|---:|
| 性能 | 7.574 TOPS | 15.527 TOPS | 2.050× |
| 延迟 | 17.720 μs | 8.644 μs | -51.22% |
| 功耗 | 23.064 W | 22.857 W | -0.90% |
| 面积 | 4.068 mm² | 4.273 mm² | +5.05% |
| 能量/task | 408.7 μJ | 197.6 μJ | -51.66% |

这说明在接近相同功耗与面积的约束下，F7 提供的 PPA 缩放可以转化为
更宽的 Vector datapath 和完整 double buffer，最终获得约 2.05 倍性能
和约 52% 任务能耗下降。

## 10. 数据可信度与限制

### 10.1 PPA 数据限制

- F7 Tensor 使用 datasheet 中的真实 F7-PKU 记录，2 GHz timing 通过；
- T7 Tensor 在 2 GHz 下没有 timing-passing 记录，当前选择 slack 最大的
  fallback 记录，slack 为 -0.012 ns；
- F7 `ara_sys` 缺少真实记录，当前按 T7 power ×0.50、area ×0.75
  派生；
- F7 SRAM 缺少真实 macro 数据，当前按 T7 power ×0.50、area ×0.50
  派生；
- SRAM 50% floorplan 利用率是架构假设，尚未由实际 floorplan 验证。

因此 F7 总功耗和面积适合作为架构探索数字，不应直接视为最终后端签核
结果。

### 10.2 性能模型限制

- Baseline 35,440 cycles 是 gem5 实测；
- Double buffer 的 26,544 cycles 是基于实测阶段时间的 overlap model；
- Vector-2x 的 17,288 cycles 包含 `vector_compute_scale=2` 和 overlap
  model；
- 当前顺序 workload 尚未真实发射跨 engine 并发；
- 下一阶段应让 workload/command scheduler 真正并发执行两个 BR
  context，再以实际 operator busy cycles 验收 26,544 和 17,288 的
  目标。

### 10.3 能耗模型限制

报告使用 PPA reported power 乘以 kernel 延迟估算任务能耗，没有使用
activity factor 驱动的 workload-specific dynamic power。能耗结果适合
比较相对趋势，绝对数值仍需门级功耗分析校准。

## 11. 推荐实施顺序

1. 保留 32-bank、35,440 cycles 作为性能 golden baseline；
2. 在 workload 和 command scheduler 中实现真实的双 BR context 并发；
3. 先验收 64-bank double buffer 是否接近 26,544 cycles；
4. 分析 softmax 中 reduce、exp、LUT 和同步开销，确认 Vector-2x 的
   真实瓶颈；
5. 实现真实双 Vector group 或等效并行执行；
6. 以实际 gem5 operator busy cycles 验收 17,288-cycle 目标；
7. 获取真实 F7 Ara、SRAM macro 和 floorplan 数据，替换当前缩放假设。

若目标是综合效率，优先选择 F7 double buffer；若目标是最高吞吐和最低
延迟，且允许更高面积，则选择 F7 Vector-2x double buffer。

## 12. 验证结果

本轮完成以下验证：

| 验证项 | 结果 |
|---|---:|
| PPA unit tests | 21/21 PASS |
| Simulation unit tests | 10/10 PASS |
| F7 baseline gem5 | PASS |
| F7 double buffer gem5 | PASS |
| F7 Vector-2x DB gem5 | PASS |
| F7 acceptance script | PASS |
| gem5 style check | PASS |
| `git diff --check` | PASS |

代表性结果目录：

- `fproject/sim/results/F7__fa_q256_kv1024_d128__20260805T032442Z/`
- `fproject/sim/results/F7-double-buffer__fa_q256_kv1024_d128__20260805T032531Z/`
- `fproject/sim/results/F7-vector2x-double-buffer__fa_q256_kv1024_d128__20260805T032615Z/`

## 13. 复现实验

PPA 对比：

```bash
./fproject/ppa_eval/run.sh
```

单独比较任意配置：

```bash
python3 fproject/ppa_eval/ppa_eval.py \
  fproject/ppa_eval/configs/t7.yaml \
  fproject/ppa_eval/configs/f7.yaml \
  --format json
```

运行 F7 baseline：

```bash
./fproject/sim/test-f7.sh
```

运行指定架构：

```bash
./fproject/sim/gem5-fa-sim.sh \
  --hardware fproject/ppa_eval/configs/f7_double_buffer.yaml \
  --task fproject/sim/configs/fa_q256_kv1024_d128.yaml
```

单元测试：

```bash
docker exec -i "${USER}.gem5" bash -lc \
  "cd /gem5 && python3 -m unittest discover \
  -s fproject/ppa_eval/tests -v"

docker exec -i "${USER}.gem5" bash -lc \
  "cd /gem5 && python3 -m unittest discover \
  -s fproject/sim/tests -v"
```

## 14. 相关文件

- `fproject/ppa_eval/model.py`：PPA 与 SRAM floorplan area 模型；
- `fproject/ppa_eval/ppa_eval.py`：PPA 输出与比较入口；
- `fproject/ppa_eval/configs/`：T7/F7 三档配置；
- `fproject/sim/fa_sim.py`：FlashAttention cycle simulation wrapper；
- `fproject/sim/buffer_pipeline_model.py`：double-buffer 流水模型；
- `fproject/sim/README.md`：性能模型说明；
- `fproject/ppa_eval/README.md`：PPA 数据和假设说明。
