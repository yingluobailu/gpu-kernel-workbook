# 09 · 双阶段标签直方图

对一段 INT32 标签统计 64 桶直方图。标签 `0–63` 对应各自的桶，其他值忽略。输入可能高度偏斜，例如绝大部分元素都落在同一个桶。先分别统计每个 tile，再归并成全局结果。

## 接口

- 输入长度 `0 ≤ n ≤ 2²²`。每个 tile 最多有 1,024 个标签，`tiles = ceil(n / 1024)`。
- `solve(labels, partials, n, tiles)` 每个 block/program 负责一个 tile。中间缓冲区是连续的 INT32 矩阵 `partials[64, tiles]`，第 `b` 桶的第 `t` 个 tile 写入 `partials[b * tiles + t]`。
- `merge(partials, histogram, tiles)` 每个 block/program 负责一个桶，输出 64 个 INT32。输入和输出缓冲区不重叠。空输入由宿主直接返回 64 个 0。
- CUDA 两阶段各 256 线程。Triton 第一阶段使用 `BLOCK=1024, BINS=64`，第二阶段使用 `TILE_COUNT`，即将 `tiles` 向上取整为 2 的幂。

示例　`[2, 2, -1, 64, 0, 63]` 只有第 0、2、63 桶非零，分别为 1、2、1。

## 提示

先让每个 block 累积自己的计数，避免所有 block 持续争用同一个全局计数器。在 CUDA 里还可以为每个 warp 分配一份 shared memory，分散不同 warp 对热点桶的争用。归并阶段让每个 block 读取同一个桶的所有 tile，观察中间布局怎样影响访存。

## 参考实现与验证

CUDA 在共享内存中使用每 warp 一份直方图。Triton 3.3 的 `tl.histogram` 没有 mask 参数，因此将无效标签和尾部放进额外的桶，只写回 64 个有效桶。两版归并都用独立 kernel，第二阶段不会读到还在更新的中间结果。

评测逐桶检查中间矩阵和最终直方图，覆盖空输入、全无效标签、单桶热点、边界标签、不足一个 tile 的尾部以及百万级输入。

```sh
python -m workbook.judge check 09 --backend both
```

[CUDA](cuda.cu) · [Triton](triton_answer.py) · [NVIDIA 双阶段直方图讲解](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
