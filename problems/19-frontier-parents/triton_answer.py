import triton
import triton.language as tl


@triton.jit
def solve(offsets, sources, state, parent, vertices, degree, BLOCK: tl.constexpr):
    dst = tl.program_id(0)
    best = tl.full((BLOCK,), 0, tl.int32) + vertices
    if tl.load(state + dst) == 0:
        first = tl.load(offsets + dst)
        end = tl.load(offsets + dst + 1)
        for tile in range(tl.cdiv(end - first, BLOCK)):
            j = first + tile * BLOCK + tl.arange(0, BLOCK)
            src = tl.load(sources + j, j < end, 0)
            active = tl.load(state + src, j < end, 0) == 1
            best = tl.minimum(best, tl.where(active, src, vertices))
    winner = tl.min(best, 0)
    tl.store(parent + dst, tl.where(winner < vertices, winner, -1))
