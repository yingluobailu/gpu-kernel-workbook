import triton
import triton.language as tl


@triton.jit
def solve(a, b, bias, out, m, n, k, TILE: tl.constexpr):
    row = tl.program_id(1) * TILE + tl.arange(0, TILE)
    col = tl.program_id(0) * TILE + tl.arange(0, TILE)
    inner = tl.arange(0, TILE)
    acc = tl.full((TILE, TILE), 0, tl.float32)
    for tile in range(tl.cdiv(k, TILE)):
        t = tile * TILE + inner
        left = tl.load(a + row[:, None] * k + t[None, :], (row[:, None] < m) & (t[None, :] < k), 0.)
        right = tl.load(b + t[:, None] * n + col[None, :], (t[:, None] < k) & (col[None, :] < n), 0.)
        acc = tl.dot(left.to(tl.float16), right.to(tl.float16), acc)
    offset = tl.load(bias + col, col < n, 0.)
    result = tl.maximum(acc + offset[None, :], 0.)
    tl.store(out + row[:, None] * n + col[None, :], result, (row[:, None] < m) & (col[None, :] < n))
