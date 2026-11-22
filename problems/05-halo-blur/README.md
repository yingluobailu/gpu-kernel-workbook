# 05 · 带 Halo 的 3×3 模糊

对连续 `rows × cols` 浮点矩阵应用以下权重，最后除以 16。边界外的采样值为 0。

```text
1 2 1
2 4 2
1 2 1
```

- 输入 `src` 和输出 `dst` 同形状、`float32`、不重叠，标量为 `rows, cols`。
- 测试数据是绝对值不超过 16 的 1/8 倍数，结果可精确比较。允许空维度。
- CUDA 使用 `16 × 16` block，Triton 使用 `TILE=16`。

示例　`[[4]]` 的四周都在域外，结果为 `[[1]]`。

**提示**　一个 `16 × 16` 的输出 tile 需要 `18 × 18` 的输入范围。可以由同一个 block 合作加载边框，让跨 tile 的邻点也按原始图像取值。

**参考实现**　CUDA 在 shared memory 中合作填入包括 halo 的 tile，再用 block barrier 同步。Triton 参考实现用 9 次带掩码的邻点加载，便于比较显式共享缓存和直接全局加载的代码与性能。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
