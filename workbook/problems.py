"""Problem contracts and deterministic, NumPy-backed judge cases."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


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
    )
}
