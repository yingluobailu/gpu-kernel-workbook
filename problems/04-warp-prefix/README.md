# 04 · Warp 内排他前缀和

将长度为 `n` 的序列按每 32 个元素分组。输出与输入等长，`dst[i]` 是组内严格位于 `i` 之前的元素之和。每组的第一个位置输出 0。

- 输入 `src`、输出 `dst` 均为 `int32`，标量为 `n ≥ 0`。
- 本题约定输入值在 `[0, 50]` 内，组内累加不会溢出。
- CUDA block 为 256 线程。Triton 用 `GROUPS=8`、每组 32 个位置。

示例　组 `[3, 1, 4, 2]` 输出 `[0, 3, 4, 8]`。第 32 号下标另起一组。

**提示**　用距离为 1、2、4、8、16 的 shuffle 构造 inclusive scan，再减掉当前元素。注意 lane 不足以向上读取时的行为。

**参考实现**　CUDA 用 `__shfl_up_sync` 做 5 轮交换；Triton 用 `tl.associative_scan` 显式指定沿组内轴操作。全局 scan 还需要处理每组之间的进位。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
