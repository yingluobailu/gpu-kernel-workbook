# 19 · BFS 扩展一层时，谁是父节点？

给定有向图的**入边**邻接表和当前一层的访问状态，求下一层新发现顶点的父节点。多个候选同时到达时选编号最小者，让同一输入的 BFS 树可以复现。

`solve(offsets, sources, state, parent, vertices, degree)`。所有参数为 int32，offsets 长 vertices+1。sources 在每行列出指向该行顶点的源点，可乱序、重复或含自环。state 和 parent 长 vertices。状态 0 表示未访问，1 表示当前前沿，2 表示更早访问过。degree 用于生成测试输入，各顶点实际入度由 offsets 给出。

仅当目标状态为 0 且存在状态为 1 的源点时，写入这些源点的最小编号，其余位置写 −1。所有读取使用同一份 state 快照。以四个顶点为例，状态为 `[1,1,0,0]`，顶点 2 的入边来自 `[1,1,0]`，顶点 3 的唯一入边来自 2，结果为 `[-1,-1,0,-1]`。顶点 2 到下一轮才有资格成为父节点。

**思路**　每个输出由一个块负责。线程跨步扫描入边，把符合条件的源点编号做最小值归约，使用 vertices 作为哨兵。CUDA 每块 128 个线程，Triton 每次并行加载 BLOCK=128 条入边。测试把更小的父节点放在长邻接表末尾，检查多次加载间是否正确保留候选。

`python -m workbook.judge check 19 --backend both`

[CUDA](cuda.cu) · [Triton](triton_answer.py) · [方向优化 BFS 论文](https://www.scottbeamer.net/pubs/beamer-sc2012.pdf)
