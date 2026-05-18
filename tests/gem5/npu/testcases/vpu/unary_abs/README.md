# VPU unary abs 测试

## 测试目的

验证 `VABS` 对 `F32` 和 `I32` 输入都能正确执行逐元素绝对值。

## 仿真程序

workload 顺序发送两条 `VABS` 命令：

- 一条覆盖 `[-3.5, -0.0, 0.0, 2.0]`
- 一条覆盖 mixed-sign `I32` 样本

## 预期行为

仿真结束时，两组结果都应与 golden 一致，并输出稳定的
`VPU_UNARY_ABS_PASS`。
