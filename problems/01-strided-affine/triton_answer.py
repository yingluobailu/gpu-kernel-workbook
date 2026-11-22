import triton
import triton.language as tl


@triton.jit
def solve(src, dst, n, stride, offset, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + offset + i * stride, i < n, other=0)
    tl.store(dst + i, 2.0 * values - 3.0, i < n)
