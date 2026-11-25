import triton
import triton.language as tl


@triton.jit
def solve(packed, scale, zero, x, out, rows, width, group_size, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    words = tl.cdiv(width, 8)
    groups = tl.cdiv(width, group_size)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tile in range(tl.cdiv(width, BLOCK)):
        col = tile * BLOCK + lane
        valid = col < width
        bits = tl.load(packed + row * words + col // 8, valid, 0).to(tl.uint32)
        code = ((bits >> ((col % 8) * 4)) & 15).to(tl.int32)
        group = row * groups + col // group_size
        z = tl.load(zero + group, valid, 0)
        s = tl.load(scale + group, valid, 0.)
        v = tl.load(x + col, valid, 0.)
        acc += (code - z).to(tl.float32) * s * v
    tl.store(out + row, tl.sum(acc, 0))
