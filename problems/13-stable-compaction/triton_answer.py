import triton
import triton.language as tl


@triton.jit
def solve(src, counts, n, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + i, i < n, other=0)
    tl.store(counts + tile, tl.sum((values > 0).to(tl.int32), axis=0))


@triton.jit
def scan(counts, offsets, total, tiles, TILE_COUNT: tl.constexpr):
    i = tl.arange(0, TILE_COUNT)
    counts_block = tl.load(counts + i, i < tiles, other=0)
    tl.store(offsets + i, tl.cumsum(counts_block, axis=0) - counts_block, i < tiles)
    tl.store(total, tl.sum(counts_block, axis=0))


@triton.jit
def scatter(src, offsets, total, values, indices, n, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + i, i < n, other=0)
    kept = value > 0
    tile_prefix = tl.cumsum(kept.to(tl.int32), axis=0) - 1
    destination = tl.load(offsets + tile) + tile_prefix
    tl.store(values + destination, value, kept)
    tl.store(indices + destination, i, kept)
    unused = (i < n) & (i >= tl.load(total))
    tl.store(values + i, 0, unused)
    tl.store(indices + i, -1, unused)
