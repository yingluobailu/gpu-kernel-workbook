import triton
import triton.language as tl


@triton.jit
def _winner(values, active, columns):
    top = tl.max(tl.where(active, values, -float("inf")), axis=0)
    index = tl.min(tl.where(active & (values == top), columns, 2147483647), axis=0)
    return top, tl.where(index == 2147483647, -1, index)


@triton.jit
def solve(scores, keep, values, indices, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * cols + columns
    active = tl.load(keep + offsets, columns < cols, other=0) != 0
    scores_tile = tl.load(scores + offsets, active, other=0)
    first, i = _winner(scores_tile, active, columns)
    second, j = _winner(scores_tile, active & (columns != i), columns)
    tl.store(values + 2 * row, first)
    tl.store(values + 2 * row + 1, second)
    tl.store(indices + 2 * row, i)
    tl.store(indices + 2 * row + 1, j)
