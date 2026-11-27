import triton
import triton.language as tl


@triton.jit
def solve(offsets, positions, owner, local, rows, queries, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(positions + i, i < queries, -1)
    valid = (p >= 0) & (p < tl.load(offsets + rows))
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), 0, tl.int32) + rows
    for _ in tl.static_range(STEPS):
        mid = (lo + hi) // 2
        active = valid & (lo < hi)
        end = tl.load(offsets + mid + 1, active, 0)
        lo = tl.where(active & (end <= p), mid + 1, lo)
        hi = tl.where(active & (end > p), mid, hi)
    base = tl.load(offsets + lo, valid, 0)
    tl.store(owner + i, tl.where(valid, lo, -1), i < queries)
    tl.store(local + i, tl.where(valid, p - base, -1), i < queries)
