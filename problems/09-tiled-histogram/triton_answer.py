import triton
import triton.language as tl


@triton.jit
def solve(labels, partials, n, tiles, BLOCK: tl.constexpr, BINS: tl.constexpr):
    tile = tl.program_id(0)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    label = tl.load(labels + offset, offset < n, other=-1)
    # Triton 3.3 has no histogram mask. Give invalid inputs a spare bin.
    valid = (label >= 0) & (label < BINS)
    counts = tl.histogram(tl.where(valid, label, BINS), 2 * BINS)
    bucket = tl.arange(0, BINS)
    tl.store(partials + bucket * tiles + tile, tl.gather(counts, bucket, 0))


@triton.jit
def merge(partials, histogram, tiles, TILE_COUNT: tl.constexpr):
    bucket = tl.program_id(0)
    i = tl.arange(0, TILE_COUNT)
    counts = tl.load(partials + bucket * tiles + i, i < tiles, other=0)
    tl.store(histogram + bucket, tl.sum(counts, axis=0))
