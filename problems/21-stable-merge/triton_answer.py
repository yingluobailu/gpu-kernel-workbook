import triton
import triton.language as tl


@triton.jit
def solve(a, b, values, origin, m, n, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    rank = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rank < m + n
    lo = tl.maximum(0, rank - n)
    hi = tl.minimum(rank, m)
    for _ in tl.static_range(STEPS):
        i = (lo + hi) // 2
        j = rank - i
        active = valid & (lo < hi)
        left = tl.load(a + i, active, 0)
        right = tl.load(b + j - 1, active, 0)
        lo = tl.where(active & (left <= right), i + 1, lo)
        hi = tl.where(active & (left > right), i, hi)
    j = rank - lo
    left = tl.load(a + lo, valid & (lo < m), 0)
    right = tl.load(b + j, valid & (j < n), 0)
    from_a = (lo < m) & ((j == n) | (left <= right))
    tl.store(values + rank, tl.where(from_a, left, right), valid)
    tl.store(origin + rank, tl.where(from_a, lo, m + j), valid)
