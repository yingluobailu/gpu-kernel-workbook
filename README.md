# GPU Kernel Workbook

Nineteen progressively harder GPU programming exercises, each with a CUDA C++ solution, a Triton solution, and the same deterministic judge. Start with memory access and warp primitives, then tackle multi-launch reductions, masked operators, sparse gathers, stream compaction and three inference kernels. Finish with sparse graph search, triangle counting and breadth-first traversal. The judge compiles and runs native GPU code without PyTorch.

一套由浅入深的 GPU 编程练习题。每题包括中文题面、输入输出协议、提示、CUDA/Triton 双参考答案和可运行测试。

| 题号 | 练习 | 核心概念 |
| --- | --- | --- |
| 01 | [带步长的仿射变换](problems/01-strided-affine/) | 寻址、尾块掩码、合并访问 |
| 02 | [带 padding 的矩阵转置](problems/02-pitched-transpose/) | 非方阵、行跨度、shared memory 与 bank conflict |
| 03 | [Warp 正数投票](problems/03-warp-vote/) | ballot、popcount、部分 warp |
| 04 | [Warp 内排他前缀和](problems/04-warp-prefix/) | shuffle、associative scan、分组 |
| 05 | [带 Halo 的 3×3 模糊](problems/05-halo-blur/) | block 协作加载、同步、边界 |
| 06 | [带稳定平局规则的行最大值](problems/06-row-argmax/) | warp 与 block 两级归约、argmax |
| 07 | [分层全局归约](problems/07-hierarchical-sum/) | 多次 kernel launch 和逐轮部分和 |
| 08 | [数值稳定的前缀遮罩 Softmax](problems/08-masked-softmax/) | 融合归约、大 logits、空有效行 |
| 09 | [双阶段标签直方图](problems/09-tiled-histogram/) | shared atomics、热点争用、逐 tile 归并 |
| 10 | [变长残差 RMSNorm](problems/10-ragged-rmsnorm/) | 融合归约、变长遮罩、尾部零值 |
| 11 | [任意位置掩码的稳定 Top‑2](problems/11-masked-top2/) | 多候选归约、平局、缺失名次 |
| 12 | [重复列与空行的 CSR SpMV](problems/12-csr-spmv/) | 稀疏访存、间接索引、负载倾斜 |
| 13 | [稳定流压缩](problems/13-stable-compaction/) | tile 计数、全局扫描、稳定 scatter |
| 14 | [带列偏置和 ReLU 的分块矩阵乘法](problems/14-bias-relu-gemm/) | 块内数据复用、矩形尾块、融合写回 |
| 15 | [分组 4 位权重的矩阵向量乘法](problems/15-packed-int4-gemv/) | 位打包、分组零点、即时解码 |
| 16 | [读取变长 KV 缓存的流式注意力](problems/16-streaming-decode/) | 在线 Softmax、跨块缩放、变长前缀 |
| 17 | [压紧数组的行号与行内位置](problems/17-csr-locate/) | CSR 偏移、重复边界、并行二分 |
| 18 | [按最小顶点计数三角形](problems/18-triangle-census/) | 有向邻接表、交集、块内归约 |
| 19 | [BFS 下一层的确定性父节点](problems/19-frontier-parents/) | 入边遍历、前沿快照、最小父节点 |

## Quick start · 快速上手

使用 64 位 Python（已验证 3.11）。CUDA 评测依赖 NVIDIA 驱动和带 NVRTC 的 CUDA Toolkit（已验证 12.6）。Triton 评测需要 NVIDIA SM 8.0+，依赖文件分平台选择 Linux Triton 和 Windows 发行包。Apple GPU、AMD GPU 和在线不可信代码执行不在此评测器的支持范围内。

```sh
python -m pip install -r requirements-triton.txt
python -m workbook.judge list
python -m workbook.judge check 03 --backend both
python -m workbook.judge check all --backend both
```

只跑 CUDA 或 CPU 测试时，安装 `requirements.txt` 即可。Windows 若无法定位 NVRTC，请检查 `CUDA_PATH`；Linux 可用 `CUDA_HOME` 或动态链接器路径指定 CUDA Toolkit。

```sh
python -m workbook.judge check all --backend cuda
python -m unittest discover -s tests -v
```

## 性能实验 · Performance lab

同一道题通过评测后，可以继续测它在真实 GPU 上的表现。四组实验分别改变访存步长、矩阵行跨度、图像长宽比和 Softmax 行宽。每组使用三种输入形状，支持 CUDA、Triton 和自己的答案。

```sh
python -m workbook.bench all
python -m workbook.bench pitch --backend cuda --submission labs/transpose_direct.cu
```

[进入实验手册，复现分块转置与直接写回的对照](labs/README.md)

## 从空白答案开始

在仓库根目录运行 `start`，生成只含必需函数签名的练习文件，再按题面实现。以下命令以第 13 题为例，模板包含计数、扫描和写回三个入口。重复运行会保留已经写过的文件。用 `--dir` 可选择另一个练习目录。

```sh
python -m workbook.judge start 13 --backend cuda
python -m workbook.judge cases 13
python -m workbook.judge check 13 --backend cuda --submission submissions/13-stable-compaction.cu
```

Triton 使用相同的流程，把 `start` 的后端改为 `triton` 即可生成 Python 文件。生成模板和查看用例只需要 Python 与 NumPy。

### 复现一次失败

用例编号从 1 开始。`cases` 按题面顺序列出形状和参数。第 13 题的第 6 例有 257 个元素，刚好跨过一个 256 元素的块。下列命令只生成并运行这一例。

```sh
python -m workbook.judge check 13 --backend cuda --submission submissions/13-stable-compaction.cu --case 6 --seed 2026
```

省略 `--case` 可在同一种子下检查整题。改变 `--seed` 会重新生成随机部分，形状和刻意构造的边界保持一致。默认种子为 343。数值判定失败时，评测器会打印含题号、后端、设备、用例编号、种子和提交路径的复测命令。在相同代码和 NumPy 环境下运行即可重建输入。[NumPy 随机数生成器文档](https://numpy.org/doc/stable/reference/random/generator.html)

## 提交自己的实现

先读题面并按缓冲区、标量和编译期参数顺序编写 `solve`。CUDA 入口声明为 `extern "C" __global__ void solve(...)`，Triton 入口使用 `@triton.jit`。多阶段题的同一文件还需提供题面列出的入口。输出缓冲区由判题器分配，所有数组在设备侧按行连续，题目 02 的 padding 由显式 `pitch` 表示。

```sh
python -m workbook.judge check 04 --backend cuda --submission my_scan.cu
python -m workbook.judge check 04 --backend triton --submission my_scan.py
```

提交仅在本机运行。只评测可信代码，Triton 提交会作为 Python 模块导入。数值不符返回退出码 1，运行或工具链错误返回 2。用 `--device` 可以指定 CUDA 设备。

## 测试覆盖

GitHub Actions 会在 Windows 和 Linux 上运行 CPU 参考测试；GPU 真机结果单独验证。

- 两个 GPU 后端共用 139 个确定性用例和 NumPy 对照实现，另有明确写出期望结果的 CPU 测试。
- 检查空输入、非整块尾部、带 NaN 的 padding、全负行、跨 warp 平局、超过百万元素的多轮归约、大 logits 和空 Softmax 前缀，以及 CSR 长短行、Top‑2 重复分值和跨 block 稳定写回。
- 整数与可精确比较的浮点例使用精确判定。全局浮点求和、Softmax、RMSNorm 和流式注意力使用误差容限。遮罩输出和空注意力头须为精确的零。多阶段题检查约定的中间缓冲区。
- 评测使用 NVRTC 或 Triton 编译出 CUBIN，再通过 CUDA Driver API 启动。Triton 版本锁定在 3.3 系列，因为本地启动适配该版本的 ABI。

Windows 实测环境为 RTX 3060 Laptop GPU（SM 8.6）、CUDA Toolkit 12.6、Python 3.11 和 Triton Windows 3.3.1。两种实现通过 278 项 GPU 检查，Compute Sanitizer memcheck 报告 0 项错误。可用以下命令复做内存检查。

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 python -m workbook.judge check all --backend both
```

参考 [NVIDIA CUDA SIMT Kernel 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)、[NVRTC 官方文档](https://docs.nvidia.com/cuda/archive/12.6.1/nvrtc/index.html)、[Triton 教程](https://triton-lang.org/main/getting-started/tutorials/)及 [Windows Triton 安装说明](https://github.com/triton-lang/triton-windows)。
