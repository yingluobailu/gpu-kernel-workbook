"""Independent CPU references for sorting, merging and segmented scans."""

import numpy as np


def sort_case(rng, rows, width):
    keys = rng.integers(-5, 6, (rows, width), dtype=np.int32)
    keys[:, ::11] = np.iinfo(np.int32).max
    keys[:, 1::11] = np.iinfo(np.int32).min
    keys[1::4] = 7
    order = np.argsort(keys, axis=1, kind="stable").astype(np.int32)
    return (keys,), (order,)


def merge_order(left, right):
    joined = np.concatenate((left, right))
    order = np.argsort(joined, kind="stable").astype(np.int32)
    return joined[order], order


def merge_case(rng, m, n):
    arrays = []
    for size in (m, n):
        values = rng.integers(-9, 10, size, dtype=np.int32)
        if size:
            values[0] = np.iinfo(np.int32).max
        if size > 1:
            values[1] = np.iinfo(np.int32).min
        arrays.append(np.sort(values))
    return tuple(arrays), merge_order(*arrays)


def segmented_sum(values, heads):
    out = np.empty_like(values)
    for r in range(len(values)):
        total = 0
        for c in range(values.shape[1]):
            if heads[r, c]:
                total = 0
            total += int(values[r, c])
            out[r, c] = total
    return out


def segment_case(rng, rows, width):
    values = rng.integers(-9, 10, (rows, width), dtype=np.int32)
    heads = (rng.random((rows, width)) < .08).astype(np.int32)
    heads[::4] = 0
    heads[1::4] = 1
    heads[2::4] = 0
    heads[2::4, 127:130] = 1
    return (values, heads), (segmented_sum(values, heads),)
