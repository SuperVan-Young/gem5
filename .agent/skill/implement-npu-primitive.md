# NPU Primitive 实现指南

## 目标

在当前 `hw_pipelined` 执行模型下，实现或优化 NPU primitive，并尽量压低
CPU 侧的指令发射开销。

## 原则

- 优先使用 `exec-only` 的 VPU 指令。在 `hw_pipelined` 模式下，上层 helper
  不应该重新拼一套带 load/store 语义的软件流水。
- 不要在热点路径里让上层 LLM primitive 反复复用下层 unary/binary/reduce
  helper。重复 lowering 和 tensor descriptor 构造通常会成为主要 CPU 开销。
- 对规则 slice 的形状只 lower 一次，然后复用命令模板。尾块或非规则形状可
  以走 fallback，不要让非常规路径拖慢主路径。
- 在 slice 循环里，只 patch 真正变化的 word。常见情况下主要是
  `DST_ADDR`、`SRC0_ADDR`、`SRC1_ADDR`。
- 多条命令共同构成一个宏算子时，应该 patch 完一条就立刻发射一条，而不是
  等整包命令都 patch 完再统一发射。
- sync 语义要保持显式。通常只有最后一条命令负责携带最终的 sync bit，而
  workload 负责外层的 `sync_wait`。
- 做 launch cost profiling 时，把钩子放在 simulator 侧。不要在 primitive
  helper 里打印日志，否则会污染真实的 CPU 执行时间。

## 推荐流程

1. 先确认执行模型约束：是否 `exec-only`、sync 归属在哪里、SPM 地址约定
   是什么。
2. 先定位热点调用栈，把 descriptor/lowering 开销和 `cmd.launchCmdAt()`
   开销分开看。
3. 为规则 slice 构建 fast path：
   - 预计算 slot span
   - 预计算 slice stride
   - 预构建 command template
   - 在循环中原地 patch 变化的地址
4. 先用单 slice case 验证正确性，再扩展到多 slice 形状。
5. 至少补一个按 shape 命名的多 slice testcase，例如
   `softmax/softmax_3x64x128xf32`。
6. 每做完一步优化就重新 profile。如果 launch gap 不再变化，优先怀疑瓶颈
   已经下沉到 launch ABI，而不是 primitive lowering 本身。

## 实用检查项

- 如果 profile 显示 `cpu_launch_request -> seu_begin` 很小，但请求之间的
  gap 很大，瓶颈通常在 CPU 侧发射软件路径，而不是硬件 dispatch。
- 如果一个 primitive 只有单 slice，那么 slice 循环优化的收益通常有限；
  需要用多 slice case 才能真正评估 fast path 是否生效。
- 优化自定义 launch 汇编时要非常谨慎。看似无害的 ABI 改动，可能会静默破坏
  staged command 的 payload。
