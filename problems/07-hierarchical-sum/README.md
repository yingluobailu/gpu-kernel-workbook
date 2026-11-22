# 07 · 分层全局归约

将连续的 `float32[n]` 拆成 512 元素一组。实现一轮归约 kernel，写出每组的部分和。判题器把部分和输入同一个 kernel，循环直到只剩一个值。

- 缓冲区为 `src, partials`，标量为本轮的 `n`。每个 block/program 写入 `partials[block_id]`。
- 缓冲区不重叠。宿主将空输入定义为 0，尾部缺少的元素也按 0 处理。
- CUDA 每 block 256 线程，每线程最多读取 2 项。Triton 使用 `BLOCK=512`。
- 测试包含 1,048,577 个元素，CPU 参考采用 FP64 累加再转换到 FP32。

示例　长度 513、全是 1 的输入，首轮输出 `[512, 1]`，下一轮输出 `[513]`。

**提示**　先在寄存器和 warp 内归约，再通过 shared memory 合并不同 warp。block 之间的阶段边界由宿主发起下一次 kernel launch。

**参考实现**　CUDA 使用两层 shuffle，Triton 在单个 program 内用 `tl.sum`。两者共用同一个宿主多轮循环。

[CUDA](cuda.cu) · [Triton](triton_answer.py)
