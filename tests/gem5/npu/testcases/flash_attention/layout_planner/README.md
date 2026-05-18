# FlashAttention Layout Planner

## 测试目的

验证 FlashAttention Stage 3 的独立 layout/scratch planner 契约，固定
`Q_i8/K_i8/K_t_i8/V_i8/scores_i32/scores_f32/m_state/l_state/p_f32/p_i8/out_i32/out_f32/scale`
的输入输出结构、地址布局和 scratch 非重叠语义。

## 仿真系统

单核 RISCV timing CPU 加默认 DRAM，仅运行纯软件 workload，不依赖
FlashAttention 后续 stage 的任何硬件路径。

## 仿真程序

workload 直接调用 `flash_attention_layout.hh` 中的 planner：

- 以固定 `seq_q/seq_k/dim/dim_v` 和 workspace 基址生成所有 buffer layout
- 校验每个 buffer 的 shape、stride、byte size 和顺序地址分配
- 校验 row-major 地址计算，以及 `K_t_i8(col, row)` 对应 `K(row, col)` 的转置地址契约
- 枚举所有 buffer 对并确认不存在重叠
- 额外验证 workspace 不足时 planner 会明确失败

## 预期行为

仿真通过时会打印：

- `FLASH_ATTENTION_LAYOUT_PLANNER_PASS ...`
- `FLASH_ATTENTION_LAYOUT_PLANNER_CONFIG_PASS`
