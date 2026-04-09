# NPU 测试 V2 目录与命名规范

本文档定义 `tests/gem5/npu` 当前 V2 结构、命名约定，以及从旧目录到新目录的映射原则。它只描述目录组织和职责边界，不包含 testcase 的具体实现细节。

## 目标结构

当前 checked-in 的 V2 结构下，`tests/gem5/npu` 以三类顶层目录组织：

- `configs/`
- `utils/`
- `testcases/`

### `configs/`

`configs/` 只放公共仿真系统构建相关的 Python 代码。

职责范围：

- 构建 gem5 仿真系统的公共 builder
- 提供多个 testcase 共享的系统拓扑 helper
- 提供公共的地址映射、组件挂接、结果采样辅助函数

不应放入的内容：

- testcase 自己的 harness 入口
- testcase 专有的 workload 代码
- testcase 的场景说明文档

### `utils/`

`utils/` 只放给仿真程序使用的公共接口。

职责范围：

- MMIO 读写 helper
- 命令打包与发射 helper
- sync / completion helper
- workload 侧稳定复用的公共头文件

不应放入的内容：

- Python 仿真系统构建代码
- testcase 运行注册逻辑
- testcase 专有数据集

### `configs/runner_common.py`

`configs/runner_common.py` 提供 testcase harness 入口复用的注册辅助函数。

职责范围：

- 解析 testcase 目录下的 `config.py` 和二进制路径
- 构造稳定命名的 regex verifier
- 从 `test.py` 路径推导 testcase 根目录并创建 testcase 本地的 `MakeTarget` / `MakeFixture`
- 注册单个 testcase 或多 scenario testcase

不应放入的内容：

- 仿真系统构建逻辑
- workload 代码
- testcase 专有的场景语义

### `testcases/`

`testcases/` 只放 testcase 本身。

组织原则：

- 先按模块分类
- 再按测试项分类
- 测试项目录尽量扁平化

每个测试项目录下的标准文件集为：

- `test.py`
- `config.py`
- `Makefile`
- `workload.c`
- `README.md`

例外说明：

- 面向 profiling 产物生成和人工验收的 testcase 可以采用独立 Makefile runner。
- 这类 testcase 可以不以 `gem5_verify_config(...)` 作为主入口，而是直接串联 workload 编译、gem5 运行、日志解析和 HTML 渲染。
- 当前仓库里没有保留这类 profiling-only 示例；如后续需要，可按该约束单独新增。

## 命名约定

### 模块目录

模块目录使用小写、稳定、可长期复用的模块名，优先按硬件单元或测试域命名。

推荐模块示例：

- `megacmdqueue/`
- `seu/`
- `vpu/`
- `dma/`
- `spm/`
- `system/`

模块名的要求：

- 表达清晰的硬件或功能边界
- 不使用临时缩写
- 不使用与单个 testcase 混淆的名字

### 测试项目录

测试项目录使用小写 `snake_case`，目录名直接作为该 testcase 的稳定标识。

示例：

- `basic_mmio/`
- `sync_indicator/`
- `4rv_sync_stress/`
- `basic/`

测试项目录名的要求：

- 直接反映测试语义
- 能独立识别 testcase
- 尽量保持稳定，不因内部实现变化而频繁改名

### 文件命名

每个 testcase 目录内部文件名固定，不再根据测试项自由变形。

固定文件名：

- `test.py`
- `config.py`
- `Makefile`
- `workload.c`
- `README.md`

## 目录职责边界

### `test.py`

`test.py` 只负责 testcase 的 harness 入口和注册，不承载仿真系统搭建逻辑。

### `config.py`

`config.py` 只负责 testcase 的仿真系统构建、运行后的结果采样，以及是否通过的判定逻辑。

### `Makefile`

`Makefile` 只负责该 testcase 的 workload 编译和本地清理，不负责目录级全局规则。

编译产物约定：

- testcase 可执行文件统一放在 testcase 目录下的 `bin/`
- 中间目标文件和依赖文件由 `Makefile` 本地管理
- 目录级忽略规则由仓库根目录统一维护，新 testcase 不应再额外添加局部 `.gitignore`

### `workload.c`

`workload.c` 只放该 testcase 的仿真程序入口和业务步骤。

### `README.md`

`README.md` 只说明该 testcase 的测试目的、仿真系统、仿真程序、预期行为，正文使用中文。

## 旧目录到新目录的映射原则

旧结构在迁移期间会保留，但新 testcase 以 `testcases/` 为唯一正式入口。映射时遵循以下原则：

1. 一个旧 testcase 应映射到一个新的测试项目录。
2. 模块名优先按硬件域归类，而不是按历史目录名机械保留。
3. 同一模块下的多个历史 testcase，如果共享同一类硬件契约，应进入同一模块目录下的不同测试项目录。

### 优先迁移案例映射

| 旧目录 | 新目录 |
| --- | --- |
| `mega/` | `testcases/megacmdqueue/basic_mmio/` |
| `sit/test_sit_sync_indicator.py` | `testcases/megacmdqueue/sync_indicator/` |
| `sit/test_sit_launch_sync_launch_sync.py` | `testcases/megacmdqueue/launch_sync_launch_sync/` |
| `4rv/` | `testcases/megacmdqueue/4rv_sync_stress/` |
| `seu/` | `testcases/seu/basic/` |

### 后续扩展原则

后续其他 testcase 迁移时，默认沿用同样的组织方式：

- 先确定模块目录
- 再确定 testcase 目录
- 再在 testcase 目录内使用固定文件集

## 迁移期间的约束

- 不在 testcase 目录内部再引入 `src/` 作为默认结构
- 不让每个 testcase 自己维护局部忽略规则来解决编译产物问题
- 不让 `Makefile` 负责跨 testcase 的全局产物清理
- 不把公共 helper 散落在 testcase 目录中
- 不把模块级公共逻辑写进 `test.py` 或 `workload.c`

## 当前状态说明

说明：

- 当前仓库实际存在的是 `utils/`，不是 `software_utils/`。
- `software_utils/` 仍然只是后续可能采用的重命名目标。
- 本文档描述的 testcase 结构、`test.py` 入口、`configs/runner_common.py` 位置均对应当前代码树。
