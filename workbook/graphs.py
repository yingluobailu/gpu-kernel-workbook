"""CSR search, triangle census and frontier expansion cases."""

import numpy as np


def csr(rows):
    offsets = np.empty(len(rows) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum([len(row) for row in rows], out=offsets[1:])
    values = np.array([value for row in rows for value in row], dtype=np.int32)
    return offsets, values


def locate_case(rng, rows, queries):
    lengths = rng.integers(0, 8, rows, dtype=np.int32)
    lengths[::3] = 0
    offsets, owners = csr([[i] * int(length) for i, length in enumerate(lengths)])
    positions = rng.integers(-2, owners.size + 2, queries, dtype=np.int32)
    probes = [-1, owners.size, 0, owners.size - 1, *offsets[:8]]
    positions[:min(queries, len(probes))] = probes[:queries]
    row = np.full(queries, -1, dtype=np.int32)
    local = row.copy()
    valid = (positions >= 0) & (positions < owners.size)
    row[valid] = owners[positions[valid]]
    local[valid] = positions[valid] - offsets[row[valid]]
    return (offsets, positions), (row, local)


def triangle_counts(forward):
    counts = np.zeros(len(forward), dtype=np.int32)
    for u, neighbors in enumerate(forward):
        for i, v in enumerate(neighbors):
            counts[u] += sum(w in forward[v] for w in neighbors[i + 1:])
    return counts


def triangle_case(rng, vertices, percent):
    forward = []
    for u in range(vertices):
        selected = np.flatnonzero(rng.integers(0, 100, vertices - u - 1) < percent) + u + 1
        if u == 0 and percent:
            selected = np.arange(1, vertices)
        forward.append(selected.tolist())
    return csr(forward), (triangle_counts(forward),)


def pull_parents(incoming, state):
    parents = np.full(len(state), -1, dtype=np.int32)
    for dst, sources in enumerate(incoming):
        if state[dst] == 0:
            parents[dst] = min((src for src in sources if state[src] == 1), default=-1)
    return parents


def frontier_case(rng, vertices, degree):
    state = rng.integers(0, 3, vertices, dtype=np.int32)
    incoming = []
    for dst in range(vertices):
        count = rng.integers(0, degree + 1)
        incoming.append(rng.integers(0, vertices, count).tolist())
    if vertices >= 4:
        state[:4] = [1, 1, 0, 0]
        incoming[2] = [1] * degree + [0]
        incoming[3] = [2]
    return (*csr(incoming), state), (pull_parents(incoming, state),)
