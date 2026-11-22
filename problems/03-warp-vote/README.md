# 03 · Warp 正数投票

把 `int32` 输入按下标分成大小为 32 的组。为每组输出一个 `int32`，表示严格大于零的元素数量。最后一个不足 32 的组只统计有效位置。

- 输入 `src[n]`，输出 `counts[ceil(n/32)]`，标量为 `n ≥ 0`。
- CUDA 每个 block 为 256 线程。Triton 每个 program 处理 `GROUPS=8` 个逻辑组。

示例　`[7, -4, 0, 3]` 得到 `[2]`。

**提示**　CUDA 尾组的无效线程应该提供 false，并跟同 warp 的其他线程一起参与投票。若先在尾部 return，再以全掩码调用 warp intrinsic，会违反参与约束。

**参考实现**　CUDA 用 `__ballot_sync` 和 `__popc`，由首 lane 写出结果。Triton 对 `[GROUPS, 32]` 的布尔矩阵按行归约，32 是题目的逻辑分组大小。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
