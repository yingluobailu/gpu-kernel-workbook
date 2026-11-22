# 02 · 带 padding 的矩阵转置

输入是一张逻辑大小为 `rows × cols` 的矩阵，每个物理行占 `pitch` 个 `float32` 元素。忽略 padding，输出连续的 `cols × rows` 矩阵，其中 `dst[c, r] = src[r * pitch + c]`。

- 缓冲依次为 `src, dst`，标量为 `rows, cols, pitch`。允许任一逻辑维度为 0。
- `pitch ≥ cols`。输入 padding 可能是 NaN，输出地址计算使用逻辑行数。输入输出不重叠。
- CUDA 的 block 为 `(32, 8, 1)`，每个 block 转置 `32 × 32` tile。Triton 的 `TILE=32`。

示例　输入物理行为 `[1,2,3,99]` 和 `[4,5,6,99]`，逻辑宽度为 3。输出为 `[[1,4],[2,5],[3,6]]`。

**提示**　连续读会让朴素转置产生跨步写。先把 tile 放在片上，再交换读出的坐标。边缘 tile 的加载掩码和写回掩码分别对应哪个坐标系？

**参考实现**　CUDA 用 `tile[32][33]` 的额外一列降低转置读取时的 bank conflict，并在加载后同步。Triton 用分块 load、`tl.trans` 和掩码 store，由编译器决定数据布局变换。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
