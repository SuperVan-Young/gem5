# PPA datasheet provenance

原始 datasheet 保存在当前项目内：

```text
source/1784445812871.xlsx
```

SHA-256：

```text
e1a8624ebc477b93651a9f6e6e4acfcda6fd96db68c9c5d17c43a543562fc2c3
```

`ppa_base_v1.csv` 是原始 XLSX 的规范化版本，`ppa_base_v1.audit.json` 记录接受及忽略的原始行。

`ppa_augmented_v1.csv` 包含全部 base 记录，以及从对应 T7 记录派生的 F7-PKU `ara_sys`：

```text
power_w = T7 power_w × 0.50
area_um2 = T7 area_um2 × 0.75
```

增强记录完整保留 `derived_from`、`power_scale` 和 `area_scale`。目标周期与 timing 结果继承源记录；源记录 timing 失败时，增强记录仍然失败。

Area 原始单位为 μm²，`area_mm2 = area_um2 / 1_000_000`。Power 使用原表唯一的总 Power 字段。
