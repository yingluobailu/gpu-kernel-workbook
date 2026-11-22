import triton
import triton.language as tl


@triton.jit
def solve(src, partials, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + i, i < n, other=0)
    tl.store(partials + tl.program_id(0), tl.sum(values, axis=0))
