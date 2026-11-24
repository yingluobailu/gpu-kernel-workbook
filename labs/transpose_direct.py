import triton
import triton.language as tl


@triton.jit
def solve(src, dst, rows, cols, pitch, TILE: tl.constexpr):
    # Flatten the input tile. Consecutive logical lanes write 'rows' elements apart.
    lane = tl.arange(0, TILE * TILE)
    r = tl.program_id(1) * TILE + lane // TILE
    c = tl.program_id(0) * TILE + lane % TILE
    active = (r < rows) & (c < cols)
    values = tl.load(src + r * pitch + c, mask=active, other=0.)
    tl.store(dst + c * rows + r, values, mask=active)
