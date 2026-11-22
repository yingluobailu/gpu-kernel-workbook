import triton
import triton.language as tl


@triton.jit
def solve(src, counts, n, GROUPS: tl.constexpr):
    group = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    i = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    values = tl.load(src + i, i < n, other=0)
    total = tl.sum((values > 0).to(tl.int32), axis=1)
    tl.store(counts + group, total, group * 32 < n)
