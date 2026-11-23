import triton
import triton.language as tl


@triton.jit
def solve(x, residual, weights, lengths, y, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    length = tl.load(lengths + row)
    active = i < length
    pos = row * cols + i
    z = tl.load(x + pos, active, other=0) + tl.load(residual + pos, active, other=0)
    ms = tl.sum(z * z, axis=0) / tl.maximum(length, 1)
    weight = tl.load(weights + i, active, other=0)
    tl.store(y + pos, z * tl.rsqrt(ms + 1e-5) * weight, i < cols)
