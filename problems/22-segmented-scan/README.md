# 22 · 跨窗口的分段前缀和

一行记录由若干连续段组成。对每个位置，写出从本段起点到该位置的和。head 为 1 时该元素开启新段。每行开头也自动开启新段。例如输入值 `[2,-1,7,3,-2]` 与标记 `[0,0,1,1,0]` 对应输出 `[2,1,7,3,1]`。

`solve(values, heads, out, rows, width)`。所有缓冲区和标量均为 int32。三张矩阵形状均为 rows × width，按行连续。标记只取 0 或 1，用例内的累加结果均可用 int32 表示。行或列可以为空。

**思路**　把一个区间表示成一对摘要，即最后一段的和，以及该区间内是否出现过段首。连接左右区间时，若右侧有段首，则保留右侧的和，否则两侧相加。标记取逻辑或。这种连接满足结合律，可以用于并行前缀扫描。

CUDA 每行一个 128 线程块，按 128 列切窗，在共享内存中通过逐次加倍的跨度做扫描。Triton 每行一个程序，对 BLOCK=128 个二元组执行 associative_scan。窗口中的某个前缀尚未遇到段首时，要加上前一窗口的末尾结果。一行扫完后清零。测试包含全行单段、每个元素独立成段、窗口边缘连续出现段首，以及 2051 列的长行。

`python -m workbook.judge check 22 --backend both`

[CUDA](cuda.cu) · [Triton](triton_answer.py) · [Triton 关联扫描接口](https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html)
