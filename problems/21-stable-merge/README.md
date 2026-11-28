# 21 · 按输出排名并行归并

将两条已按升序排列的 int32 流合并，同时写出每个结果在拼接输入中的位置。键相等时，A 中的元素优先，同一来源内部保持原顺序。A 为 `[1,3,3]`，B 为 `[1,2,3]` 时，输出键为 `[1,1,2,3,3,3]`，来源位置为 `[0,3,4,1,2,5]`。

`solve(a, b, values, origin, m, n)`。所有参数都是 int32，A 和 B 分别长 m 和 n，两个输出均长 m+n。来源位置 0 到 m−1 属于 A，其余属于 B。任一输入可以为空，INT_MIN 和 INT_MAX 都是有效键。

**思路**　要得到排名 k 的元素，先求前 k 个结果从 A 和 B 分别取了多少个。设取自 A 的数量为 i，取自 B 的数量为 j=k−i，二分区间是 `[max(0,k−n), min(k,m)]`。若 A[i] ≤ B[j−1]，说明前缀还应多取一个 A 的元素。找到分界后，比较下一对候选即可写出唯一的结果。分别判断剩余长度可避免把极值误当哨兵。

CUDA 每块 128 个线程，每线程解一个排名。Triton 每个程序处理 BLOCK=128 个排名，使用 STEPS 轮有掩码的二分。第 6 例包含两条等长流与大量重复键，可以同时检查分界和来源顺序。

`python -m workbook.judge check 21 --backend both`

[CUDA](cuda.cu) · [Triton](triton_answer.py) · [Merge Path 的分界方法](https://moderngpu.github.io/merge.html)
