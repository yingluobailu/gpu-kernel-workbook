import triton
import triton.language as tl


@triton.jit
def solve(keys, order, rows, width, BLOCK: tl.constexpr):
    base = tl.program_id(0) * width
    i = tl.arange(0, BLOCK)
    key = tl.load(keys + base + i, i < width, 0)
    biased = key.to(tl.uint32) ^ 0x80000000
    pair = (biased.to(tl.uint64) << 32) | i.to(tl.uint64)
    pair = tl.where(i < width, pair, 0xffffffffffffffff)
    sorted_pair = tl.sort(pair, descending=False)
    tl.store(order + base + i, sorted_pair.to(tl.int32), i < width)
