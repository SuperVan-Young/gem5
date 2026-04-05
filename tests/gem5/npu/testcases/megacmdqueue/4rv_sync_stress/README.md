# MegaCmdQueue 四核同步压力测试

## 测试目的

验证共享 `MegaCmdQueue` 多输入端口在四个 CPU 并发提交场景下的稳定性，重点覆盖 sync wait、sync set、端口隔离和下游 `SEU` 命令完成计数。

## 仿真系统

该 testcase 使用 4 个 CPU、默认物理内存、一个 `MegaCmdQueue` 和一个基础 `SEU`。
`MegaCmdQueue` 配置为 4 个输入端口，每个 CPU 使用自己的命令端口窗口和同一个同步指示器 MMIO 基址。

## 仿真程序

每个 CPU 在自己的端口上循环若干轮：
先提交一个 sync wait 命令，再提交一个合法的最小 VPU/SEU 执行命令，然后立即写对应的同步指示器释放等待命令。
循环内保留少量软件 jitter，让四个核的提交节奏产生交错。

## 预期行为

仿真结束时，退出原因应正确，`MegaCmdQueue` 队列占用应为 0，`SEU` 队列占用应为 0，`SEU` 完成命令数应等于 `4 * rounds`。
`config.py` 需要输出稳定的通过标记 `MEGACMDQUEUE_4RV_SYNC_STRESS_PASS`。
