# 10 · 变长残差 RMSNorm

输入行的有效长度各不相同。对每行的前 `k=lengths[row]` 个元素先做残差相加 `z=x+residual`，再输出 `z × rsqrt(mean(z²)+1e-5) × weights[col]`。均方只统计有效元素。尾部和空行输出精确的零。

- 缓冲区依次为 FP32 的 `x, residual, weights`、INT32 的 `lengths`、FP32 输出 `y`。行张量为 `[rows, cols]`，`weights` 形状为 `[cols]`。
- 标量为 `rows, cols`，允许零行，`1 ≤ cols ≤ 4096`，且 `0 ≤ k ≤ cols`。有效输入和权重绝对值不超过 16，无效输入可为 NaN。
- CUDA 每行一个 256 线程 block。Triton 的 `BLOCK` 为向上取整到 2 的幂的列数。

示例　`x=[1,2,NaN]`、`residual=[2,2,NaN]`、`weights=[1,2,9]`、`k=2` 时，结果为 `[3,8,0]/sqrt(12.5+1e-5)`。

**提示**　把加载掩码放在读操作上。归约结果参与全行的输出计算，CUDA 需要在写回前同步并广播结果。

**对照**　CUDA 用 warp shuffle 和 shared memory 归约，再分段重读有效数据。Triton 在一个 program 中用掩码加载、`tl.sum` 和 `tl.rsqrt`，复用已经加载的 tile。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
