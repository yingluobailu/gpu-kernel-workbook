import triton
import triton.language as tl


@triton.jit
def solve(src, dst, rows, cols, pitch, TILE: tl.constexpr):
    x = tl.program_id(0) * TILE + tl.arange(0, TILE)
    y = tl.program_id(1) * TILE + tl.arange(0, TILE)
    values = tl.load(src + y[:, None] * pitch + x[None, :], (y[:, None] < rows) & (x[None, :] < cols), other=0)
    tl.store(dst + x[:, None] * rows + y[None, :], tl.trans(values), (x[:, None] < cols) & (y[None, :] < rows))
