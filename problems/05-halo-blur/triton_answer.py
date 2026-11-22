import triton
import triton.language as tl


@triton.jit
def _pixel(src, y, x, rows, cols):
    return tl.load(src + y * cols + x, (y >= 0) & (y < rows) & (x >= 0) & (x < cols), other=0)


@triton.jit
def solve(src, dst, rows, cols, TILE: tl.constexpr):
    y = tl.program_id(1) * TILE + tl.arange(0, TILE)[:, None]
    x = tl.program_id(0) * TILE + tl.arange(0, TILE)[None, :]
    center = _pixel(src, y, x, rows, cols)
    axial = (
        _pixel(src, y - 1, x, rows, cols) + _pixel(src, y + 1, x, rows, cols)
        + _pixel(src, y, x - 1, rows, cols) + _pixel(src, y, x + 1, rows, cols)
    )
    corner = (
        _pixel(src, y - 1, x - 1, rows, cols) + _pixel(src, y - 1, x + 1, rows, cols)
        + _pixel(src, y + 1, x - 1, rows, cols) + _pixel(src, y + 1, x + 1, rows, cols)
    )
    tl.store(dst + y * cols + x, (4.0 * center + 2.0 * axial + corner) / 16.0, (y < rows) & (x < cols))
