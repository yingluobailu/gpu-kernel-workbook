import triton
import triton.language as tl


@triton.jit
def _join(left_sum, left_head, right_sum, right_head):
    return tl.where(right_head != 0, right_sum, left_sum + right_sum), left_head | right_head


@triton.jit
def solve(values, heads, out, rows, width, BLOCK: tl.constexpr):
    base = tl.program_id(0) * width
    lane = tl.arange(0, BLOCK)
    carry = 0
    for tile in range(tl.cdiv(width, BLOCK)):
        c = tile * BLOCK + lane
        val = tl.load(values + base + c, c < width, 0)
        head = tl.load(heads + base + c, c < width, 0)
        sums, seen = tl.associative_scan((val, head), 0, _join)
        sums += tl.where(seen == 0, carry, 0)
        tl.store(out + base + c, sums, c < width)
        carry = tl.sum(tl.where(lane == BLOCK - 1, sums, 0), 0)
