# 15 · 分组 4 位权重的矩阵向量乘法

模型权重以 4 位无符号码存储，每行再按列分组，分别保存缩放系数和零点。直接从压缩权重计算一行的点积，显存中的权重始终保持压缩形式。

## 缓冲区协议

`solve(packed, scale, zero, x, out, rows, width, group_size)`。标量为 int32，`group_size ≥ 1`。所有缓冲区按行连续。

| 缓冲区 | 数据与形状 |
| --- | --- |
| packed | int32 位容器，形状为 rows × ceil(width/8)。每行单独起一个新字。列 c 的码占该字第 4(c mod 8) 位起的四位，低位在前。 |
| scale、zero | 分别为 float32 和 int32，形状均为 rows × ceil(width/group_size)。零点范围为 0 到 15。 |
| x、out | float32 输入向量长度为 width，输出长度为 rows。 |

令 `g = c // group_size`。每一列对第 r 行的贡献是 `(code[r,c] - zero[r,g]) * scale[r,g] * x[c]`。将它们相加。分组可能跨过字边界，同一个字也可能包含多个组。最后一字的高位可能全为 1。宽度为零时，按行写出零。测试使用二进制可精确表示的小分数和有界点积，结果逐元素精确比较。

例如一行三个码 `[1,2,15]` 可用字 `0xFFFFFF21` 表示。组大小为 2，零点为 `[2,8]`，缩放系数为 `[0.5,0.25]`。解出的三个权重是 `[-0.5,0,1.75]`，与 `[2,-1,4]` 点积得到 6。

## 怎样分摊工作

CUDA 每行分配 128 个线程。线程取出一整个无符号字，逐个提取其中的有效码，块内归约各线程的小计。Triton 每个程序处理一行，`BLOCK=128` 个列位置同时提取码、查找本组参数，再沿列归约。将窗口扩大到 1025 列可覆盖线程的第二轮取字及不足一字的尾部。CPU 对照直接使用打包前的码计算，另有逐位解码测试核对数据格式。

```sh
python -m workbook.judge check 15 --backend both
```

[CUDA 解法](cuda.cu) · [Triton 解法](triton_answer.py) · [ONNX 对缩放系数和零点的定义](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html)
