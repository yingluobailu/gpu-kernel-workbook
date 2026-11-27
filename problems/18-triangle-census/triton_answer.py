import triton
import triton.language as tl


@triton.jit
def solve(offsets, adj, counts, vertices, percent, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    u = tl.program_id(0)
    first = tl.load(offsets + u)
    end = tl.load(offsets + u + 1)
    lane = tl.arange(0, BLOCK)
    subtotal = tl.full((BLOCK,), 0, tl.int32)
    for j in range(first, end):
        v = tl.load(adj + j)
        start_v = tl.load(offsets + v)
        end_v = tl.load(offsets + v + 1)
        for tile in range(tl.cdiv(end_v - start_v, BLOCK)):
            p = start_v + tile * BLOCK + lane
            w = tl.load(adj + p, p < end_v, -1)
            lo = tl.full((BLOCK,), 0, tl.int32) + j + 1
            hi = tl.full((BLOCK,), 0, tl.int32) + end
            for _ in tl.static_range(STEPS):
                mid = (lo + hi) // 2
                active = lo < hi
                key = tl.load(adj + mid, active, 0)
                lo = tl.where(active & (key < w), mid + 1, lo)
                hi = tl.where(active & (key >= w), mid, hi)
            found = tl.load(adj + lo, lo < end, -1)
            subtotal += ((p < end_v) & (found == w)).to(tl.int32)
    tl.store(counts + u, tl.sum(subtotal, 0))
