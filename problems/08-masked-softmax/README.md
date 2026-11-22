# 08 · 数值稳定的前缀遮罩 Softmax

每行具有独立的有效前缀长度 `k = lengths[row]`。在前 `k` 个元素上做 Softmax，尾部输出严格的 0。若 `k=0`，整行输出 0。

- 缓冲区依次为 `src[rows,cols]`、`lengths[rows]`、`dst[rows,cols]`，元素类型分别为 FP32、INT32、FP32。
- 标量为 `rows, cols`。允许零行，`1 ≤ cols ≤ 4096`，`0 ≤ k ≤ cols`。有效 logits 有限，无效尾部可包含 NaN。
- CUDA 每行一个 256 线程 block。Triton 每行一个 program，`BLOCK` 为列数向上取整到 2 的幂。

示例　`[1000, 1000, NaN]` 与 `k=2` 得到 `[0.5, 0.5, 0]`。

**提示**　先求最大值，再计算 `exp(x - max)`。空前缀要单独定义最大值和分母，否则 `-inf - (-inf)` 会产生 NaN。

**参考实现**　CUDA 复用两级归约做行 max 和指数和，复用 shared memory 前同步。Triton 对有效前缀加载，融合两次归约和归一化写入。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
