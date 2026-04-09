# Golden Models

本目录存放 testcase 使用的软件参考模型。

约束：

- `.hh` 只放接口声明，保持可读且便于快速浏览。
- `.cc` 只放实现，避免 testcase 把 expected 计算继续堆在本地。
- golden model 只负责“给定输入，计算预期输出”，不负责命令发射，也不负责等待或断言。

当前已拆分的模型：

- `vpu_unary.hh/.cc`
- `vpu_elemwise.hh/.cc`
- `vpu_reduce.hh/.cc`
