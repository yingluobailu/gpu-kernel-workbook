# 06 · 带稳定平局规则的行最大值

给定按行连续的 `float32[rows, cols]`，输出每行最大值和第一次出现该最大值的列索引。

- 缓冲区依次为 `src, values, indices`，后两者分别为 `float32[rows]`、`int32[rows]`。
- 标量为 `rows, cols`。允许 `rows=0`，`1 ≤ cols ≤ 4096`，有效输入为有限数。
- CUDA 每行一个 256 线程 block。Triton 每行一个 program，`BLOCK` 为列数向上取整到 2 的幂。

示例　`[[-8, -2, -2], [-5, -5, -5]]` 输出最大值 `[-2, -5]` 和索引 `[1, 0]`。

**提示**　归约的候选对象是 `(值, 列号)`。值相同的时候始终选择较小列号。padding 要使用最大归约的中性值。

**参考实现**　CUDA 分别在 warp 和 block 内合并候选。Triton 先用 `tl.max` 找最大值，再对等于该值的有效列下标使用 `tl.min`。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
