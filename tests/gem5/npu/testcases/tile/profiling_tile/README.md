# Tile 级 NPU Profiling 测试

## 测试目的

验证 NPU profiling 输出链路是否完整可用：包括 VPU/DMA 宏指令 begin/end 时间戳输出、日志解析、以及 HTML 甘特图渲染。

## 仿真系统

该 testcase 构建一个包含 1 个 CPU 输入端口、1 个 DMA、2 个独立 VPU、1 个独立 LUT、1 个 Scratchpad Memory 的 tile 级系统。单 CPU 会顺序发射跨 DMA/VPU 的混合命令流，用于稳定生成多 axis 的 profiling 结果。

## 仿真程序

workload 先通过 DMA 将 DRAM 数据搬运到 SPM，再由 `VPU[0]` 执行线性缩放，由 `VPU[1]` 执行 softmax，最后再通过 DMA 将结果回写到 DRAM。为避免 SEU 执行时间被命令提交开销淹没，这个 profiling testcase 保持 helper 兼容的 16 元素端口布局，使用更高的 CPU 时钟来压缩命令发射空洞，并把 VPU/LUT timing 维持在 ns / sub-us 量级。运行过程中会生成 `NPUProfile` 调试日志，并由 Python 脚本解析成 JSON，再渲染成单文件 HTML 甘特图。

## 预期行为

执行 `make html` 后应生成：

- `out/run/npu_profile.log`
- `out/profile_events.json`
- `out/profile_gantt.html`

其中 HTML 页面应能展示不同 SEU axis 的宏指令时间线，支持“完整时长”适配、事件聚焦、放大/缩小，并在鼠标悬停时显示宏指令字段详情。程序员可通过端口转发在浏览器中打开该 HTML 手工验收 profiling 结果。
