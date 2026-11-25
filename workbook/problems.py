"""Problem contracts and deterministic, NumPy-backed judge cases."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
HISTOGRAM_BINS = 64
HISTOGRAM_TILE = 1024
COMPACTION_TILE = 256


def ceil_div(size, tile):
    return (size + tile - 1) // tile


@dataclass(frozen=True)
class Case:
    dims: tuple
    inputs: tuple
    expected: tuple

    @property
    def label(self):
        return " × ".join(map(str, self.dims))


@dataclass(frozen=True)
class Problem:
    key: str
    slug: str
    title: str
    case_factory: object
    shapes: tuple
    cuda_block: tuple = (256, 1, 1)
    triton_warps: int = 4

    @property
    def path(self):
        return ROOT / "problems" / f"{self.key}-{self.slug}"

    def cases(self):
        for index, dims in enumerate(self.shapes):
            inputs, outputs = self.case_factory(np.random.default_rng(343 + index), *dims)
            yield Case(dims, inputs, outputs)

    def launch_config(self, dims):
        """Return the logical grid and Triton constexpr parameters."""
        n = dims[0]
        if self.key == "01":
            return (ceil_div(n, 256), 1, 1), {"BLOCK": 256}
        if self.key in {"02", "05"}:
            tile = 32 if self.key == "02" else 16
            return (ceil_div(dims[1], tile), ceil_div(n, tile), 1), {"TILE": tile}
        if self.key in {"03", "04"}:
            groups = 256 // 32
            return (ceil_div(n, groups * 32), 1, 1), {"GROUPS": groups}
        if self.key == "07":
            return (ceil_div(n, 512), 1, 1), {"BLOCK": 512}
        if self.key == "09":
            grid = (ceil_div(n, HISTOGRAM_TILE), 1, 1)
            return grid, {"BLOCK": HISTOGRAM_TILE, "BINS": HISTOGRAM_BINS}
        if self.key == "12":
            return (n, 1, 1), {"BLOCK": 256}
        if self.key == "13":
            return (ceil_div(n, COMPACTION_TILE), 1, 1), {"BLOCK": COMPACTION_TILE}
        if self.key == "14":
            return (ceil_div(dims[1], 16), ceil_div(n, 16), 1), {"TILE": 16}
        if self.key == "15":
            return (n, 1, 1), {"BLOCK": 128}
        if self.key == "16":
            return (n, 1, 1), {"TOKENS": 32}
        columns = dims[1]
        return (n, 1, 1), {"BLOCK": 1 << (columns - 1).bit_length()}


def fixed(rng, shape):
    return rng.integers(-100, 101, size=shape).astype(np.float32) / 8


def strided_affine(src, n, stride, offset):
    return (2 * src[offset:offset + n * stride:stride] - 3).astype(np.float32)


def affine_case(rng, n, stride, offset):
    src = fixed(rng, offset + (n - 1) * stride + 1 if n else offset)
    return (src,), (strided_affine(src, n, stride, offset),)


def pitched_transpose(src, cols):
    return np.ascontiguousarray(src[:, :cols].T)


def transpose_case(rng, rows, cols, pitch):
    src = fixed(rng, (rows, pitch))
    src[:, cols:] = np.nan
    return (src,), (pitched_transpose(src, cols),)


def warp_vote(src):
    return np.array([np.count_nonzero(src[i:i + 32] > 0) for i in range(0, src.size, 32)], dtype=np.int32)


def vote_case(rng, n):
    src = rng.integers(-3, 4, size=n, dtype=np.int32)
    src[:32] = 1
    src[32:64] = 0
    return (src,), (warp_vote(src),)


def warp_prefix(src):
    out = np.empty_like(src)
    for i in range(0, src.size, 32):
        part = src[i:i + 32]
        out[i:i + 32] = np.cumsum(part, dtype=np.int32) - part
    return out


def prefix_case(rng, n):
    src = rng.integers(0, 51, size=n, dtype=np.int32)
    src[:32] = 1
    return (src,), (warp_prefix(src),)


def halo_blur(src):
    rows, cols = src.shape
    padded = np.pad(src, 1)
    out = np.zeros_like(src)
    for dy, weights in enumerate(((1, 2, 1), (2, 4, 2), (1, 2, 1))):
        for dx, weight in enumerate(weights):
            out += weight * padded[dy:dy + rows, dx:dx + cols]
    return out / 16


def blur_case(rng, rows, cols):
    src = fixed(rng, (rows, cols))
    return (src,), (halo_blur(src),)


def row_argmax(src):
    first = np.argmax(src, axis=1).astype(np.int32)
    return src[np.arange(len(src)), first], first


def argmax_case(rng, rows, cols):
    src = fixed(rng, (rows, cols)) - 32
    if rows:
        src[0, cols // 3] = 64
        src[0, -1] = 64
    if rows > 1:
        src[1] = -10
    return (src,), row_argmax(src)


def hierarchical_sum(src):
    return np.array([src.astype(np.float64).sum()], dtype=np.float32)


def sum_case(rng, n):
    src = fixed(rng, n)
    return (src,), (hierarchical_sum(src),)


def masked_softmax(src, lengths):
    out = np.zeros_like(src)
    for row, length in enumerate(lengths):
        if length:
            values = src[row, :length].astype(np.float64)
            exponent = np.exp(values - np.max(values))
            out[row, :length] = exponent / exponent.sum()
    return out


def softmax_case(rng, rows, cols):
    src = fixed(rng, (rows, cols))
    lengths = rng.integers(0, cols + 1, size=rows, dtype=np.int32)
    for row in range(rows):
        if row % 4 == 0:
            lengths[row] = 0
        elif row % 4 == 1:
            lengths[row] = 1
            src[row, 0] = -1_000
        elif row % 4 == 2:
            lengths[row] = cols
            src[row] += 1_000
        src[row, lengths[row]:] = np.nan
    return (src, lengths), (masked_softmax(src, lengths),)


def histogram_counts(labels):
    valid = labels[(labels >= 0) & (labels < HISTOGRAM_BINS)]
    return np.bincount(valid, minlength=HISTOGRAM_BINS).astype(np.int32)


def partial_histogram(labels):
    counts = np.empty((HISTOGRAM_BINS, ceil_div(labels.size, HISTOGRAM_TILE)), dtype=np.int32)
    for tile, start in enumerate(range(0, labels.size, HISTOGRAM_TILE)):
        counts[:, tile] = histogram_counts(labels[start:start + HISTOGRAM_TILE])
    return counts


def histogram_case(rng, n, distribution):
    labels = rng.integers(-4, HISTOGRAM_BINS + 4, size=n, dtype=np.int32)
    if distribution == 1:
        labels[:] = HISTOGRAM_BINS - 1
    elif distribution == 2:
        labels[:] = -1
        labels[::2] = HISTOGRAM_BINS
    elif distribution == 3:
        labels[rng.random(n) < .96] = 17
        labels[::97] = -2_147_483_648
        labels[1::97] = 2_147_483_647
    return (labels,), (histogram_counts(labels),)


def ragged_rmsnorm(x, residual, weights, lengths):
    result = np.zeros_like(x)
    for row, length in enumerate(lengths):
        if length:
            z = x[row, :length].astype(np.float64) + residual[row, :length]
            result[row, :length] = z * weights[:length] / np.sqrt(np.mean(z * z) + 1e-5)
    return result


def rmsnorm_case(rng, rows, cols):
    x = fixed(rng, (rows, cols))
    residual = fixed(rng, (rows, cols))
    weights = fixed(rng, cols)
    lengths = rng.integers(0, cols + 1, size=rows, dtype=np.int32)
    lengths[::4] = 0
    lengths[1::4] = 1
    lengths[2::4] = cols
    residual[2::8] = -x[2::8]
    invalid = np.arange(cols)[None, :] >= lengths[:, None]
    x[invalid] = residual[invalid] = np.nan
    return (x, residual, weights, lengths), (ragged_rmsnorm(x, residual, weights, lengths),)


def masked_top2(scores, keep):
    results = np.full((len(scores), 2), -np.inf, dtype=np.float32)
    indices = np.full((len(scores), 2), -1, dtype=np.int32)
    for row, (values, mask) in enumerate(zip(scores, keep)):
        columns = np.flatnonzero(mask)
        # Stable sort leaves smaller columns first on ties.
        winners = columns[np.argsort(-values[columns], kind="stable")[:2]]
        results[row, :len(winners)] = values[winners]
        indices[row, :len(winners)] = winners
    return results, indices


def top2_case(rng, rows, cols):
    scores = fixed(rng, (rows, cols)) - 32
    keep = rng.integers(0, 2, size=(rows, cols), dtype=np.int32)
    keep[::5] = 0
    keep[1::5] = 0
    keep[1::5, -1] = 1
    keep[2::5] = 1
    scores[2::5] = -2
    keep[3::5, [0, cols // 2, cols - 1]] = 1
    scores[3::5, [0, cols // 2, cols - 1]] = 50
    scores[keep == 0] = np.nan
    return (scores, keep), masked_top2(scores, keep)


def csr_spmv(offsets, columns, weights, vector):
    result = np.empty(offsets.size - 1, dtype=np.float32)
    for row, (first, last) in enumerate(zip(offsets, offsets[1:])):
        result[row] = np.dot(weights[first:last].astype(np.float64), vector[columns[first:last]])
    return result


def csr_case(rng, rows, cols):
    lengths = rng.integers(0, min(cols, 40) + 1, size=rows, dtype=np.int32)
    if cols:
        for row, size in enumerate((0, 1, 31, 257, 4097)[:rows]):
            lengths[row] = size if size <= cols + 2 else min(size, cols)
    offsets = np.empty(rows + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    columns = rng.integers(0, max(cols, 1), size=int(offsets[-1]), dtype=np.int32)
    weights = fixed(rng, columns.shape)
    vector = fixed(rng, cols)
    return (offsets, columns, weights, vector), (csr_spmv(offsets, columns, weights, vector),)


def stable_compact(source):
    positions = np.flatnonzero(source > 0)
    values = np.zeros_like(source)
    indices = np.full_like(source, -1)
    values[:len(positions)] = source[positions]
    indices[:len(positions)] = positions
    return values, indices, np.array([len(positions)], dtype=np.int32)


def compaction_counts(source):
    start = np.arange(0, source.size, COMPACTION_TILE)
    return np.add.reduceat((source > 0).astype(np.int32), start, dtype=np.int32)


def compaction_case(rng, n, distribution):
    source = rng.integers(-8, 9, size=n, dtype=np.int32)
    if distribution == 1:
        source[:] = 7
    elif distribution == 2:
        source[:] = 0
        source[::2] = -2_147_483_648
    elif distribution == 3:
        source[:COMPACTION_TILE] = -1
        source[COMPACTION_TILE:2 * COMPACTION_TILE] = 9
        source[::113] = 2_147_483_647
    return (source,), stable_compact(source)


def biased_relu_gemm(a, b, bias):
    return np.maximum(a.astype(np.float64) @ b + bias, 0).astype(np.float32)


def gemm_case(rng, m, n, k):
    a = rng.integers(-8, 9, size=(m, k)).astype(np.float32) / 4
    b = rng.integers(-8, 9, size=(k, n)).astype(np.float32) / 4
    bias = rng.integers(-8, 9, size=n).astype(np.float32) / 4
    return (a, b, bias), (biased_relu_gemm(a, b, bias),)


def int4_case(rng, rows, width, group_size):
    words = ceil_div(width, 8)
    groups = ceil_div(width, group_size)
    # Independent oracle uses dense codes before packing; unused nibbles are set to 15.
    codes = rng.integers(0, 16, (rows, width), dtype=np.uint32)
    padded = np.full((rows, words * 8), 15, dtype=np.uint32)
    padded[:, :width] = codes
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    packed = np.bitwise_or.reduce(padded.reshape(rows, words, 8) << shifts, axis=2).view(np.int32)
    zeros = rng.integers(0, 16, (rows, groups), dtype=np.int32)
    scales = rng.integers(1, 9, (rows, groups)).astype(np.float32) / 8
    vector = rng.integers(-8, 9, width).astype(np.float32) / 4
    group = np.arange(width) // group_size
    weights = (codes.astype(np.float64) - zeros[:, group]) * scales[:, group]
    return (packed, scales, zeros, vector), ((weights @ vector).astype(np.float32),)


def decode_attention(query, keys, values, lengths):
    output = np.zeros_like(query)
    for head, length in enumerate(lengths):
        if length:
            scores = keys[head, :length].astype(np.float64) @ query[head] / np.sqrt(32.)
            weights = np.exp(scores - scores.max())
            output[head] = weights @ values[head, :length] / weights.sum()
    return output


def decode_case(rng, heads, capacity):
    query = fixed(rng, (heads, 32)) / 8
    keys = fixed(rng, (heads, capacity, 32)) / 8
    values = fixed(rng, (heads, capacity, 32)) / 8
    pattern = (0, 1, capacity, 31, 32, 33, capacity)
    lengths = np.array([min(capacity, pattern[h % len(pattern)]) for h in range(heads)], dtype=np.int32)
    for head, length in enumerate(lengths):
        if head % 7 == 2 and length:
            # The last token changes the running maximum; earlier mass still matters.
            query[head] = 0
            query[head, 0] = 8
            keys[head, :length, 0] = 1020
            keys[head, length - 1, 0] = 1024
        if head % 7 == 3:
            query[head] = 0  # Equal scores should average all valid value vectors.
        if head % 7 == 6:
            query[head] = 0
            query[head, 0] = 8
            keys[head, :length, 0] -= 1024  # Unshifted exponentials underflow.
        keys[head, length:] = np.nan
        values[head, length:] = np.nan
    return (query, keys, values, lengths), (decode_attention(query, keys, values, lengths),)


PROBLEMS = {
    problem.key: problem
    for problem in (
        Problem("01", "strided-affine", "Strided affine transform", affine_case,
                ((0, 1, 0), (1, 7, 2), (255, 1, 0), (257, 3, 1), (8193, 7, 4), (262145, 1, 0))),
        Problem("02", "pitched-transpose", "Pitched matrix transpose", transpose_case,
                ((0, 5, 9), (7, 0, 5), (1, 35, 42), (31, 33, 40), (63, 65, 70), (513, 1025, 1033)),
                cuda_block=(32, 8, 1)),
        Problem("03", "warp-vote", "Warp positive vote", vote_case,
                ((0,), (1,), (31,), (32,), (33,), (257,), (16387,))),
        Problem("04", "warp-prefix", "Warp-local exclusive scan", prefix_case,
                ((0,), (1,), (31,), (32,), (33,), (257,), (65539,))),
        Problem("05", "halo-blur", "Zero-padded halo blur", blur_case,
                ((0, 7), (1, 1), (1, 17), (17, 1), (17, 33), (129, 257)),
                cuda_block=(16, 16, 1)),
        Problem("06", "row-argmax", "Row max with first-index ties", argmax_case,
                ((0, 5), (1, 1), (4, 31), (4, 33), (4, 257), (4, 1025), (3, 4096))),
        Problem("07", "hierarchical-sum", "Hierarchical full-vector sum", sum_case,
                ((0,), (1,), (33,), (511,), (512,), (513,), (262145,), (1048577,))),
        Problem("08", "masked-softmax", "Stable prefix-masked softmax", softmax_case,
                ((0, 2), (4, 1), (9, 17), (11, 33), (7, 257), (8, 1025), (8, 4096))),
        Problem("09", "tiled-histogram", "Two-stage tiled histogram", histogram_case,
                ((0, 0), (1, 1), (33, 2), (1023, 0), (1024, 1), (1025, 0), (65567, 3), (1048577, 3))),
        Problem("10", "ragged-rmsnorm", "Ragged residual RMSNorm", rmsnorm_case,
                ((0, 7), (4, 1), (4, 17), (8, 255), (7, 257), (8, 1025), (5, 4095), (9, 4096))),
        Problem("11", "masked-top2", "Stable masked top-2", top2_case,
                ((0, 7), (5, 1), (9, 2), (9, 31), (11, 33), (9, 257), (9, 1025), (9, 4096))),
        Problem("12", "csr-spmv", "CSR matrix-vector multiply", csr_case,
                ((0, 0), (4, 0), (1, 1), (6, 7), (23, 127), (25, 257), (257, 8191))),
        Problem("13", "stable-compaction", "Stable positive-value compaction", compaction_case,
                ((0, 0), (1, 1), (31, 2), (255, 1), (256, 0), (257, 3), (65537, 3), (262145, 3), (1048576, 3))),
        Problem("14", "bias-relu-gemm", "Tiled GEMM with column bias and ReLU", gemm_case,
                ((0, 7, 3), (5, 0, 9), (3, 7, 0), (1, 1, 1), (15, 17, 16),
                 (17, 31, 33), (63, 65, 127), (129, 97, 257)), cuda_block=(16, 16, 1)),
        Problem("15", "packed-int4-gemv", "Groupwise packed INT4 GEMV", int4_case,
                ((0, 17, 3), (5, 0, 7), (3, 1, 1), (7, 8, 3), (9, 9, 7),
                 (13, 127, 19), (17, 257, 32), (23, 1025, 63)), cuda_block=(128, 1, 1)),
        Problem("16", "streaming-decode", "Prefix-limited streaming decode attention", decode_case,
                ((0, 11), (4, 0), (7, 1), (7, 31), (7, 32), (7, 33), (9, 129), (14, 513)),
                cuda_block=(32, 1, 1)),
    )
}
