# 13 · 稳定流压缩

将一维 INT32 输入的正数按原顺序放到前面，并返回它们的原始下标及保留数量。两个输出数组长度均为 `n`；未使用尾部的值固定为 0、下标固定为 `-1`。

示例　`[-2,7,0,7,-1,3]` 输出值 `[7,7,3,0,0,0]`、下标 `[1,3,5,-1,-1,-1]`，保留数量为 3。

## 三阶段接口

| 入口 | 缓冲区顺序与标量 | 调度 |
| --- | --- | --- |
| `solve` | `src, counts, n` | 每个 block/program 统计一个 256 元素 tile 的正数数目 |
| `scan` | `counts, offsets, total, tiles` | 一个 block/program 对计数做全局排他扫描并写一个 `total` |
| `scatter` | `src, offsets, total, values, indices, n` | 每个 block/program 写回一个 tile 的正数及相应下标，并覆盖未使用尾部 |

所有缓冲区和标量均为 INT32，`0 ≤ n ≤ 2²⁰`，`tiles=ceil(n/256)`。空输入由宿主返回空数组和计数 0。CUDA 三阶段均为 256 线程。Triton 的 `solve` 与 `scatter` 使用 `BLOCK=256`，`scan` 的 `TILE_COUNT` 为向上取整到 2 的幂的 tile 数。

**提示**　输出下标等于所属 tile 的全局偏移加上该元素在 tile 内的排他排名。全局原子计数拿到的序号无法保证元素原顺序。

**对照**　CUDA 通过 warp shuffle 和 shared memory 拼接 block scan，跨 256 个计数块的前缀用 carry 传播。Triton 对 tile 使用 `tl.cumsum`，并把全部 tile 计数装进一个 program 做小规模全局扫描。判题器检查每个 tile 的计数、偏移，以及最终结果。

[CUDA](cuda.cu) · [Triton](triton_answer.py) · [NVIDIA 并行扫描与流压缩](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
