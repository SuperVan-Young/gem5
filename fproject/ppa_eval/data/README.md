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

原表第 27、29 行给出 `ara_sys` 的 CPU 和 Other 分量。规范化数据把
两者之和保存为 `shared_*` 字段，供“共享部分只计一份、仅扩展
4-lane Vector/SRAM”模型使用。由于原表只在 `no_macro` 记录旁提供
组件拆分，`with_macro` 的 lane-group PPA 是用整机值减去共享分量
得到的推导值。

`ppa_augmented_v1.csv` 包含全部 base 记录，以及从对应 T7 记录派生的 F7-PKU `ara_sys`：

```text
power_w = T7 power_w × 0.50
area_um2 = T7 area_um2 × 0.75
```

增强记录完整保留 `derived_from`、`power_scale` 和 `area_scale`。目标周期与 timing 结果继承源记录；源记录 timing 失败时，增强记录仍然失败。

Area 原始单位为 μm²，`area_mm2 = area_um2 / 1_000_000`。Power 使用原表唯一的总 Power 字段。

## SRAM 数据

`sram_v1.csv` 独立保存 SRAM macro 数据，不修改或混入原始 compute
datasheet。每个 macro 的组织形式为 `16384 × 32 bit`，即 65536 byte
（64 KiB），可工作在 2 GHz：

```text
T7 static power = 0.063735 mW
T7 area         = 23249.160000 μm²

F7 static power = T7 × 0.50
F7 area         = T7 × 0.50
```

F7 记录的 `derived_from`、`static_power_scale` 和 `area_scale` 保留
上述派生关系。默认每 SM 的 256 KiB 配置使用 4 个 macro。SRAM 只提供
静态功耗数据；系统报告不会将其描述为 SRAM 动态功耗。
