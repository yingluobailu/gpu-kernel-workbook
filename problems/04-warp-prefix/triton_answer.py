import triton
import triton.language as tl


@triton.jit
def solve(src, dst, n, GROUPS: tl.constexpr):
    group = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    i = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    values = tl.load(src + i, i < n, other=0)
    inclusive = tl.associative_scan(values, 1, _add)
    tl.store(dst + i, inclusive - values, i < n)


@triton.jit
def _add(left, right):
    return left + right
