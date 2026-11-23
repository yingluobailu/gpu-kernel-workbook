# 11 · 任意位置掩码的稳定 Top‑2

对每行独立选择分数最高的两个有效列。分数相同优先选较小列号，两个名次必须来自不同列。缺失名次的值为负无穷，列号为 `-1`。

- 输入 FP32 `scores[rows,cols]` 和 INT32 二值掩码 `keep[rows,cols]`。
- 输出 FP32 `values[rows,2]` 和 INT32 `indices[rows,2]`。缓冲区顺序为 `scores, keep, values, indices`，标量为 `rows, cols`。
- 允许零行，`1 ≤ cols ≤ 4096`。有效位置的分数有限，无效位置可为 NaN。CUDA 线程数为 256，Triton 用向上取整到 2 的幂的 `BLOCK`。

示例　分数 `[5,9,5,5]` 和掩码 `[1,0,1,1]` 输出值 `[5,5]` 和列号 `[0,2]`。

**提示**　如果每个线程只保留自己的第一名，全局第二名可能被丢掉。归约的状态可以保存两个候选。

**对照**　CUDA 在寄存器里保存两个候选，用 warp shuffle 和共享内存逐级合并。Triton 先选第一名，再屏蔽其列号做第二次归约。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
