import triton
import triton.language as tl


@triton.jit
def solve(row_offsets, column_indices, weights, x, y, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(row_offsets + row)
    end = tl.load(row_offsets + row + 1)
    offset = tl.arange(0, BLOCK)
    terms = tl.full((BLOCK,), 0.0, tl.float32)
    for base in range(start, end, BLOCK):
        index = base + offset
        active = index < end
        columns = tl.load(column_indices + index, active, other=0)
        value = tl.load(x + columns, active, other=0)
        weight = tl.load(weights + index, active, other=0)
        terms += weight * value
    tl.store(y + row, tl.sum(terms, axis=0))
