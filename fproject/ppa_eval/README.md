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
- Vector/Tensor 实例数和所选 datasheet 记录；
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
    README.md
  tests/
```

运行依赖 Python 3 和 PyYAML。gem5 Docker 环境已经具备这些依赖；其他环境可按 `requirements.txt` 准备。

## 评估口径

- 系统频率：2 GHz。
- Tensor：128×128 仿真阵列折算为 16 个 `Mesh_BOTH_32x32`。
- Vector：512 B/cycle FP32 数据通路为 128 FP32 ops/cycle；每个 4-lane `ara_sys` 提供 4 ops/cycle，因此需要 32 个。
- 每个 `ara_sys` 完整计入 CPU、RVV 和 L1Cache。
- Vector/Tensor 版图利用率默认均为 50%。
- flow 默认 `DC-Innovus`，`ara_sys` 默认 `no_macro`。
- Area 的原始单位是 μm²，最终同时换算并显示 mm²。
- Power 使用 datasheet 中唯一的总 Power。

F7 缺少真实 `ara_sys` 数据，因此 augmented datasheet 按以下规则生成：

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
