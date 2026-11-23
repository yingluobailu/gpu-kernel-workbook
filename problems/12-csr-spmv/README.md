# 12 · 重复列与空行的 CSR SpMV

给定零基 CSR 矩阵与稠密向量 `x`，计算 `y=A×x`。行的条目数可差很多，列索引可乱序或重复。重复列的每个权重都参与累加，空行输出零。

- 缓冲区依次为 `row_offsets[rows+1]`、`column_indices[nnz]`、`weights[nnz]`、`x[cols]`、`y[rows]`，前两者为 INT32，其余为 FP32。
- 标量为 `rows, cols`。偏移首项为 0、末项为 `nnz`、单调不减。列号在 `[0, cols)`。允许零行或零列。
- 每行最多 4,097 个条目，测试权重和向量项为绝对值不超过 16 的 1/8 倍数。CUDA 每行一个 256 线程 block，Triton 每行一个 program，`BLOCK=256`。

示例　偏移 `[0,3,3,4]`、列号 `[2,0,2,1]`、权重 `[2,1,-1,4]`、`x=[3,5,7]`，结果为 `[10,0,20]`。

**提示**　分别追踪三个访存流：偏移决定边界，列号和权重连续读取，向量 `x` 需要按列号间接读取。

**对照**　CUDA 让同一 block 跨步遍历一行，然后做两级归约。Triton 分块累加每个 lane 的乘积，循环结束时做一次行归约。行长度分布会影响两种方法的负载均衡。

[CUDA](cuda.cu) · [Triton](triton_answer.py) · [NVIDIA CSR 格式](https://docs.nvidia.com/cuda/cusparse/)
