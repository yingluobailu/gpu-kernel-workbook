import triton
import triton.language as tl


@triton.jit
def solve(src, lengths, dst, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    length = tl.load(lengths + row)
    active = column < length
    x = tl.load(src + row * cols + column, active, other=-float("inf"))
    maximum = tl.where(length > 0, tl.max(x, axis=0), 0.0)
    numerator = tl.exp(x - maximum)
    divisor = tl.where(length > 0, tl.sum(numerator, axis=0), 1.0)
    tl.store(dst + row * cols + column, numerator / divisor, column < cols)
