# GPU 性能实验

先从转置开始。把一个 `1024 × 2048` 的矩阵转过来，每种实现都要写出同样的 2,097,152 个元素。可以比较同一套输入下，线程直接跨行写回和先在块内重新排列数据的实测差别。

从仓库根目录运行以下命令。环境依赖见[安装说明](../README.md#quick-start--快速上手)。性能实验使用 CUDA Graph 和计时事件，已在 Windows、CUDA Toolkit 12.6 和 RTX 3060 Laptop 上验证。

```sh
python -m workbook.bench pitch
python -m workbook.bench pitch --backend cuda --submission labs/transpose_direct.cu
python -m workbook.bench pitch --backend triton --submission labs/transpose_direct.py
```

同一台 RTX 3060 Laptop 上的一轮采样如下。矩阵大小固定，pitch 以 float32 元素为单位。表中是预热后 21 次单次执行的中位数，单位为微秒。

| pitch | CUDA 直接写回 | CUDA 分块转置 | Triton 直接写回 | Triton 分块转置 |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 213.0 | 64.3 | 209.9 | 62.5 |
| 2049 | 210.9 | 63.5 | 212.0 | 65.5 |
| 2080 | 210.9 | 67.6 | 210.8 | 65.5 |

在 pitch 为 2048 的这组中，两份直接写回实现都用了约 210 微秒。比较 CUDA 的两个版本，沿着相邻线程追踪一次读、一次写，找出地址从哪一步开始相隔整行。再看 Triton 的二维 tile 布局，检查 `tl.trans` 前后的坐标。[直接写回 CUDA](transpose_direct.cu) · [直接写回 Triton](transpose_direct.py) · [分块版本及题面](../problems/02-pitched-transpose/)

## 四个问题

| 实验 | 保持一致 | 改变什么 | 可以观察什么 |
| --- | --- | --- | --- |
| `stride` | 1,048,576 个输出，起始偏移 0 | 相邻输入相隔 1、4、8 个元素 | 每增加一级步长，处理同样多的输出要多花多久？输入占用的内存也随步长增长。 |
| `pitch` | 逻辑矩阵和输出形状 | 每行分配 2048、2049、2080 个元素 | 只改一列物理 padding，影响是否超过多次测量的波动？ |
| `stencil` | 2,097,152 个像素，同一个 3×3 算子 | 256×8192、1024×2048、8192×256 | 改变行跨度和边界与内部像素的比例，两种实现的耗时如何变化？ |
| `softmax` | 2,097,152 个有效输入，每行全有效 | 把同样多的元素排成 256、1024、4096 列 | 行变长时，每行的归约工作增加，同时可并行处理的行数减少。哪个形状最快？ |

```sh
python -m workbook.bench all
python -m workbook.bench softmax --warmup 10 --samples 51 --json
python -m workbook.bench stride --backend cuda --device 0 --submission my_affine.cu
```

## 怎样读数

- 每次复用同一组显存缓冲区。程序先编译、上传输入，再捕获含开始事件、一次内核启动和结束事件的 CUDA Graph。每次回放都等待完成后读取 GPU 时间。计时区间排除了编译、内存传输和 Python 提交，包含设备端事件节点的开销。[CUDA 事件文档](https://docs.nvidia.com/cuda/archive/12.6.1/cuda-driver-api/group__CUDA__EVENT.html)
- 每个形状先回放一次并与 NumPy 对照，再预热（默认 5 次），随后采样（默认 21 次），最后再次核对输出。自定义答案还会先跑该题的完整边界用例。正确性检查失败时程序返回 1 并停止生成报告。
- `median us` 是一次内核执行的中位耗时，`p10` 和 `p90` 描述这组原始采样的分布。`Gelem/s` 用逻辑输出数除以中位耗时，不同算子的每个输出所需工作量各异。实际 DRAM 流量需要硬件性能计数器。
- `--json` 将完整报告写到标准输出，包括 GPU、计算能力、操作系统、Python 版本、实现文件名、随机种子、输入维度和全部原始样本。比较自己的改动时保留前后两份报告，在同一设备和电源模式下交替复测。笔记本温度、时钟和并行负载都会影响数值。
