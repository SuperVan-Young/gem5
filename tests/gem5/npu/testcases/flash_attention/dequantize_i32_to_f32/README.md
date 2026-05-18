## 测试目的

验证 FlashAttention Stage 2 的独立解量化 helper，覆盖 `int32 -> f32`
输出和 scale 元数据回传，为后续算子复用提供稳定 reference。

## 仿真系统

使用 gem5 NPU testcase 默认的 RISC-V SE 模式最小系统，仅运行静态用户态
程序，不引入后续 stage 的 fused 路径。

## 仿真程序

程序构造一组 `int32` 累加结果，调用 `primitiveDequantizeI32ToF32()`，
检查：
- 解量化后的浮点值与给定 scale 相符；
- `intToInputScale`、`inputToIntScale`、`absMax` 元数据正确；
- `scale = 0` 时输出稳定清零且元数据可回传。

## 预期行为

workload 打印 `FLASH_ATTENTION_DEQUANTIZE_I32_TO_F32_PASS`，配置脚本打印
`FLASH_ATTENTION_DEQUANTIZE_I32_TO_F32_CONFIG_PASS`，测试通过。
