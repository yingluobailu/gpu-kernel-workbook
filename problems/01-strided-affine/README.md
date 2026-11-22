# 01 · 带步长的仿射变换

给定一段 `float32` 缓冲区，从 `offset` 开始每隔 `stride` 个元素取一个值，写成紧凑的 `dst`。令 `dst[i] = 2 * src[offset + i * stride] - 3`。

- 输入 `src`，输出 `dst[n]`，标量依次为 `n, stride, offset`（`int32`）。
- `n ≥ 0`，`stride ≥ 1`，`offset ≥ 0`，有效地址的计算结果在 `int32` 范围内。
- CUDA 使用 256 线程的 block。Triton 的编译参数为 `BLOCK=256`。

示例 `src=[99, 2, 99, 5, 99, -1]`，`n=3, stride=2, offset=1`，结果为 `[1, 7, -5]`。

**提示**　先给每个输出下标分配一个线程或元素，给尾块加掩码。比较 `stride=1` 和 `stride=7` 时，一个 warp 的输入地址覆盖了多少内存段。

**参考实现**　CUDA 使用线程上界检查；Triton 分别在 load 和 store 上传递掩码。两者的输出写入连续地址，输入合并访问的程度取决于步长和对齐。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
