# 16 · 读取变长 KV 缓存的流式注意力

解码一个新 token 时，每个注意力头只有一个查询向量，历史键和值已经缓存在显存。逐头返回注意力输出。一次启动中流式遍历该头的有效前缀，避免给每个历史 token 分配分数或概率缓冲区。

## 输入与输出

`solve(q, keys, values, lengths, out, heads, capacity)`。q 和 out 的形状为 `(heads,32)`，keys 和 values 的形状为 `(heads,capacity,32)`。这些缓冲区均为连续 float32。lengths 是每个头一个 int32 的有效长度，范围为 0 到 capacity，两个标量也为 int32。

对于有效 token t，计算 `s[t] = dot(q, keys[t]) / sqrt(32)`。用这些分数的 Softmax 概率对 values 加权。每个头独立归一化。空前缀的 32 个输出必须全部为零。无效缓存位置可能为 NaN。非空头与 float64 对照结果比较，绝对容差为 3e-5，相对容差为 1e-4。

例如 q 全零且长度为 2。若两条有效 value 向量分别全为 −1 和 5，两条权重均为 1/2，输出的每个分量都是 2。即便第三条缓存全是 NaN，结果也应相同。

## 一趟遍历如何归一化

用 m 记录已见最大分数，用 z 记录相对于 m 的指数和，用 u 记录同样缩放的加权向量。初始取 m 为负无穷，z 和 u 为零。读入一个非空分块后，令新最大值为 M，并令 `a = exp(m - M)`。按下式更新。

```text
w[t] = exp(s[t] - M)
z    = a * z + sum(w)
u    = a * u + sum(w[t] * value[t])
m    = M
```

结束后写回 u/z。最后一块出现更大分数时，旧分子和旧分母都需要乘以 a。测试包括极大的正分数、极小的负分数、均匀权重，以及有效长度 31、32、33。

CUDA 的块含一个 warp，32 条 lane 分担特征维度，用 shuffle 求查询和键的点积，逐 token 更新。Triton 每个头启动一个程序，`TOKENS=32`，一次归并一个分块。两份答案使用不同的合并粒度，可以用同一个 NumPy 对照验证。

```sh
python -m workbook.judge check 16 --backend both
```

[CUDA 解法](cuda.cu) · [Triton 解法](triton_answer.py) · [FlashAttention 原论文中的分块 Softmax](https://arxiv.org/abs/2205.14135)
