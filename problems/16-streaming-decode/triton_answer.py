import triton
import triton.language as tl


@triton.jit
def solve(q, keys, values, lengths, out, heads, capacity, TOKENS: tl.constexpr):
    head = tl.program_id(0)
    d = tl.arange(0, 32)
    token = tl.arange(0, TOKENS)
    length = tl.load(lengths + head)
    query = tl.load(q + head * 32 + d)
    numerator = tl.full((32,), 0, tl.float32)
    largest = -float("inf")
    mass = 0.
    for tile in range(tl.cdiv(length, TOKENS)):
        t = tile * TOKENS + token
        index = (head * capacity + t[:, None]) * 32 + d[None, :]
        k = tl.load(keys + index, t[:, None] < length, 0.)
        v = tl.load(values + index, t[:, None] < length, 0.)
        scores = tl.sum(k * query[None, :], 1) * 0.1767766952966369
        scores = tl.where(t < length, scores, -float("inf"))
        next_max = tl.maximum(largest, tl.max(scores, 0))
        carry = tl.exp(largest - next_max)
        weights = tl.exp(scores - next_max)
        numerator = numerator * carry + tl.sum(weights[:, None] * v, 0)
        mass = mass * carry + tl.sum(weights, 0)
        largest = next_max
    tl.store(out + head * 32 + d, numerator / tl.where(length > 0, mass, 1.))
