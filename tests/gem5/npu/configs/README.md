# NPU Configs Public Layer

`tests/gem5/npu/configs/` 是 NPU testcase 的公共 Python 基础设施层。这里既放 testcase `config.py` 复用的仿真配置 helper，也放 testcase `test.py` 复用的 harness 注册 helper；但这里不放 workload 代码。

## 目录职责

- [`npu_test_system.py`](./npu_test_system.py)：低层系统 builder，负责 gem5 的基础组件、地址映射和 SimObject 挂接。
- [`npu_test_common.py`](./npu_test_common.py)：面向 testcase `config.py` 的公共 helper，负责常见 build recipe、结果采样和结果验证辅助。
- [`runner_common.py`](./runner_common.py)：面向 testcase `test.py` 的公共 helper，负责路径解析、fixture 构造、verifier 构造和 `gem5_verify_config(...)` 注册。

## 推荐使用方式

后续 testcase 的 `config.py` 建议遵循同一条流水线：

1. `build_m5_system(args)` 构建仿真系统，返回 builder 或封装后的 context。
2. `collect_simulation_result(builder, args)` 采样结果，形成稳定的字典或结构体。
3. `verify_simulation_result(builder, args)` 仅做判断，不再混入系统搭建逻辑。

## 公共能力

### Build

`npu_test_common.py` 提供适合基础 testcase 的构建辅助，重点覆盖：

- 单 CPU 基础系统
- 多 CPU 基础系统
- `MegaCmdQueue` / `SEU` / `SPM` / `sync` / `DRAM` 的常见映射组合

### Map

公共映射 helper 会把常见 CPU / process 到设备地址的映射动作收拢起来，避免 testcase 重复写：

- `builder.map_cmdq(...)`
- `builder.map_seu(...)`
- `builder.map_spm(...)`
- `builder.map_sync(...)`
- `builder.map_dram(...)`

基础 testcase 可以直接使用 `map_single_cpu_regions(...)` 和 `map_multi_cpu_regions(...)`，也可以让 build helper 自动完成默认映射。

### Collect

公共采样函数会把常见运行态整理成稳定的 key/value 结构，便于 `config.py` 打印摘要和做验证。

### Verify

`config.py` 里的验证逻辑应尽量保持薄，只比较公共 helper 采样出来的结果是否满足 testcase 预期，不直接编码系统搭建细节。

## 约束

- 不把 testcase 专有语义写进公共 helper。
- 不把复杂系统级场景硬塞进基础 helper。
- 不要求 testcase 重写已有 include 路径或 harness 逻辑。
- 不把 workload 侧头文件或 C/C++ 代码放进 `configs/`。
