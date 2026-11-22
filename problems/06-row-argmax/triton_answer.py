import triton
import triton.language as tl


@triton.jit
def solve(src, values, indices, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x = tl.load(src + row * cols + column, column < cols, other=-float("inf"))
    largest = tl.max(x, axis=0)
    first = tl.min(tl.where((column < cols) & (x == largest), column, 2147483647), axis=0)
    tl.store(values + row, largest)
    tl.store(indices + row, first)
