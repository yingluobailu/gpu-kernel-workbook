# 17 · 压紧数组的位置属于哪一行？

图的邻接表把不同长度的行首尾相接。给出 CSR 行偏移和一批打乱顺序的物理位置，同时写出它们的行号与行内位置。连续空行会产生相同偏移。

`solve(offsets, positions, owner, local, rows, queries)` 的缓冲区和标量均为 int32。offsets 长 rows+1，从 0 开始且单调不减。positions、owner 和 local 均长 queries。对合法位置 p，找到唯一满足 `offsets[r] ≤ p < offsets[r+1]` 的 r，写出 r 和 p−offsets[r]。负位置以及大于等于总长度的位置，两项都写 −1。

例如 offsets 为 `[0,0,2,2,5,5]`，查询 `[2,0,4,5,-1]`，行号应为 `[3,1,3,-1,-1]`，行内位置为 `[0,0,2,-1,-1]`。

**思路**　查找首个严格大于 p 的行结束位置。遇到相等值时继续向右查，才能跨过重复偏移。CUDA 每线程处理一个查询，块大小 128。Triton 每个程序并行处理 BLOCK=128 个查询，STEPS 为 rows 的二进制位数，各 lane 单独维护搜索区间。CPU 从逐元素展开的行号表直接索引，验证边界、空输入和尾块。

`python -m workbook.judge check 17 --backend both`

[CUDA](cuda.cu) · [Triton](triton_answer.py) · [NumPy 对二分边界的定义](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html)
