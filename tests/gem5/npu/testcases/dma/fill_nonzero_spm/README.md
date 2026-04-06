# DMA fill_nonzero_spm

## 测试目的
验证 DMA 的 fill 模式可以把 SPM 目标区域整行写成固定非零字节值。这个 case 单独确认 SPM 目标空间的非零填充。

## 仿真系统
该 testcase 使用单 CPU 的 timing 系统，挂接 MegaCmdQueue、DMAUnit、一段可访问的 DRAM 地址空间和一块 SPM。
测试只发射当前目录对应的那一类 DMA 宏命令，不和其他 fill、transpose 或 queue 场景混跑。

## 仿真程序
程序先把目标 SPM 区域预填成另一组非零值，再发射一条 fill 命令，把目标地址指向 SPM 中的一整条 64B 行，fill 值固定为 `0x3c`。`workload.c` 只调用 `scenario_fill_nonzero_spm()`。
共享头 `tests/gem5/npu/utils/dma_functional_cases.hh` 只保存可复用的数据准备和命令构造逻辑；当前目录的 `workload.c` 负责把入口固定到这一条场景。

## 预期行为
gem5 应看到 `DMA_SUMMARY scenario=fill_nonzero_spm cmds=1 reads=0 writes=1 ...`，说明 DMA 成功把目标 SPM 区域整体改写。随后应看到 `DMA_SCENARIO_PASS=fill_nonzero_spm`。
