# End-to-End NPU PPA Evaluation

这是一个可独立使用的 T7/F7 PPA 验收项目。目录内包含原始 datasheet、规范化及增强数据、T7/F7 配置、计算程序、测试和一键运行入口。

## 一键运行

在仓库根目录执行：

```sh
./fproject/ppa_eval/run.sh
```

脚本会直接打印：

- T7 和 F7 的总 Power、总 Area；
- F7 相对 T7 的 Power/Area 改善比例；
- Vector/Tensor/SRAM 实例数和所选 datasheet 记录；
- timing 是否通过，以及是否使用了 timing fallback。

输出 JSON：

```sh
./fproject/ppa_eval/run.sh --format json
```

## 目录内容

```text
fproject/ppa_eval/
  run.sh                       # 最终验收入口
  ppa_eval.py                  # 单配置计算及双配置比较 CLI
  model.py                     # 选择、算力换算、PPA 汇总
  normalize_snapshot.py        # 原始 XLSX -> base datasheet
  augment_datasheet.py         # base -> augmented datasheet
  requirements.txt
  configs/
    t7.yaml
    f7.yaml
  data/
    source/1784445812871.xlsx  # 原始 datasheet 快照
    ppa_base_v1.csv
    ppa_base_v1.audit.json
    ppa_augmented_v1.csv
    sram_v1.csv
    README.md
  tests/
```

运行依赖 Python 3 和 PyYAML。gem5 Docker 环境已经具备这些依赖；其他环境可按 `requirements.txt` 准备。

## 评估口径

- 系统频率：2 GHz。
- Tensor：4 个 `Mesh_BOTH_32x32` 提供 4096 MAC/cycle；FMA
  按乘、加 2 ops 计，因此在 2 GHz 下提供 16.384 TOPS。
- Vector：512 B/cycle FP32 数据通路为 128 FP32 elements/cycle。
  Ara 每 lane 每周期处理一个 64-bit packet，即 2 个 FP32 elements；
  每个 4-lane group 提供 8 elements/cycle，因此需要 16 个 group。
- Vector FMA 同样按 2 ops/element 计，峰值为 0.512 TFLOPS。
- Vector 采用 2 GHz timing 通过的 `with_macro`（SRAM）记录。
- `ara_sys` 只保留 1 份共享 CPU 和 Other；扩展吞吐时只复制
  4-lane Vector/SRAM 部分。共享部分来自原表第 27、29 行：
  T7 为 0.0784 W、182267.71 μm²。
- 原表没有 `with_macro` 的组件拆分。因此模型从所选 `with_macro`
  总 PPA 中扣除上述 CPU/Other，余量视为一个 4-lane Vector/SRAM
  group。这是显式记录在输出中的推导口径。
- SRAM 使用 4096×128（64 KiB）macro。为了保留类似 GPU shared
  memory 的独立寻址能力，一个物理 macro 计为一个逻辑 bank，每个
  bank 对外提供 4 B/cycle，而不是把 128-bit word 等效为 4 个 bank。
- Baseline 使用 32 banks，提供单方向 128 B/cycle；256 KiB 容量只需
  4 个 macro，因此最终由 32-bank 组织约束主导，物理容量为 2 MiB。
- Double-buffer 方案使用 64 banks 和两个分址的 32-bank buffer
  context，提供单方向 256 B/cycle，物理容量为 4 MiB。
- Vector-2x double-buffer 方案把 4-lane Vector/SRAM group 从 16 组
  增加到 32 组，并使用 96 banks：Tensor 独占 32 banks（128
  B/cycle），双倍 Vector 独占 64 banks（256 B/cycle）。总 SRAM
  带宽为 384 B/cycle，物理容量为 6 MiB。
- 每个 macro 按同拍 1R1W 建模，读写可重叠。
- SRAM Power 使用 spec 给出的每 macro 报告值，不再描述为静态功耗。
- Vector/Tensor 版图利用率默认均为 50%。
- flow 默认 `DC-Innovus`，`ara_sys` 默认 `with_macro`。
- Area 的原始单位是 μm²，最终同时换算并显示 mm²。
- Power 使用 datasheet 中唯一的总 Power。

SRAM 使用独立的 `data/sram_v1.csv`。T7 单个 4096×128 macro 的
Power 为 0.041590 W、面积为 20923.128 μm²；F7 的 Power 和面积均按
T7 的 50% 派生。SRAM 数据没有 flow、利用率或 timing slack，模型
只校验系统频率不超过其 2 GHz 工作能力。

`configs/f7.yaml` 中的 `simulation.spm.size_bytes=8388608` 是 gem5
性能仿真的功能性工作区，不是 PPA 计费容量；PPA 读取
`memory.sram.capacity_bytes`、`bank_count` 和 `bank_width_bytes`。

F7 缺少真实 `ara_sys` 数据，因此 augmented datasheet 对总值及共享
CPU/Other 分量使用相同规则生成：

```text
F7 ara_sys Power = T7 ara_sys Power × 0.50
F7 ara_sys Area  = T7 ara_sys Area  × 0.75
```

缩放发生在独立的数据增强阶段，主计算器中没有 F7 特判。增强行记录了源记录和缩放比例。

## Timing fallback

选择时先精确匹配 PDK、flow、利用率、模块、variant 和目标频点，并优先采用 timing 通过的数据。

当前 T7 数据存在 timing 未通过项。若不存在 timing 通过的候选，初版会选择同频点中 slack 最大的一条，同时明确输出：

```text
timing_met=False
timing_fallback=True
```

失败记录不会被伪装成 timing 通过。

## 重新生成 datasheet

所有输入都在本目录中，无需访问在线腾讯表格：

```sh
python3 fproject/ppa_eval/normalize_snapshot.py \
  --input fproject/ppa_eval/data/source/1784445812871.xlsx \
  --output fproject/ppa_eval/data/ppa_base_v1.csv \
  --audit-output fproject/ppa_eval/data/ppa_base_v1.audit.json

python3 fproject/ppa_eval/augment_datasheet.py \
  --input fproject/ppa_eval/data/ppa_base_v1.csv \
  --output fproject/ppa_eval/data/ppa_augmented_v1.csv
```

## 自测

按仓库要求在 gem5 Docker 中运行：

```sh
docker exec -i "${USER}.gem5" bash -lc \
  "cd /gem5 && python3 -m unittest discover \
  -s fproject/ppa_eval/tests -v"
```
