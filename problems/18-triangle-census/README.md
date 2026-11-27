# 18 · 每个顶点负责多少个三角形？

把无向简单图的每条边按顶点编号从小到大存储，每个邻接表严格递增。统计以 u 为最小顶点的三角形数，所有输出相加就是整张图的三角形数。

`solve(offsets, adj, counts, vertices, percent)`。缓冲区与标量均为 int32。offsets 长 vertices+1，末项是 adj 的长度。邻接区间采用左闭右开约定。counts 长 vertices，孤立顶点写 0。percent 是生成测试图时的采样参数，邻接关系由 offsets 和 adj 给出。

四个顶点两两相连时，offsets 为 `[0,3,5,6,6]`，adj 为 `[1,2,3,2,3,3]`。三个三角形以 0 为最小顶点，一个以 1 为最小顶点，输出 `[3,1,0,0]`。

**思路**　固定边 u→v，枚举 v 的邻居 w，并在 u 的列表中 v 之后的部分二分查找 w。若命中，必有 u<v<w，每个三角形恰好计一次。CUDA 每块 64 个线程负责一个 u，线程处理不同的 w，再用共享内存求和。Triton 用 BLOCK=64 的窗口扫描 v 的邻居，各 lane 分别二分，STEPS 为 vertices 的二进制位数。测试包含空图、完全图和高度数顶点。可以从第 4 个用例复现上面的四顶点完全图。

`python -m workbook.judge check 18 --backend both`

[CUDA](cuda.cu) · [Triton](triton_answer.py) · [NVIDIA CSR 存储说明](https://docs.nvidia.com/cuda/cusparse/storage-formats.html)
