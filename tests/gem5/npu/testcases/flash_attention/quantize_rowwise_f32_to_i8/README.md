## 测试目的

验证 FlashAttention Stage 2 的独立逐行量化 helper，覆盖 `f32 -> int8`
输出、逐行 scale 元数据回传，以及软件 clamp/saturate 收尾行为。

## 仿真系统

使用 gem5 NPU testcase 默认的 RISC-V SE 模式最小系统，只运行用户态静态
workload，不依赖后续 stage 的算子链路。

## 仿真程序

程序构造两行浮点输入，调用 `primitiveQuantizeRowwiseF32ToI8()`，检查：
- 正常比例缩放后的量化值；
- 大值在 `[-127, 127]` 范围内饱和；
- 每行 `inputToIntScale`、`intToInputScale`、`absMax` 元数据正确返回；
- 全零行的 scale 元数据稳定为零。

## 预期行为

workload 打印 `FLASH_ATTENTION_QUANTIZE_ROWWISE_F32_TO_I8_PASS`，配置脚本
打印 `FLASH_ATTENTION_QUANTIZE_ROWWISE_F32_TO_I8_CONFIG_PASS`，测试通过。
