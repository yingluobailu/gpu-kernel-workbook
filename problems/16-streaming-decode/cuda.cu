// One warp handles one head. Each lane owns one feature of Q and the numerator.
extern "C" __global__ void solve(const float* q, const float* keys, const float* values,
                                 const int* lengths, float* out, int heads, int capacity) {
    const int head = blockIdx.x, lane = threadIdx.x;
    const int length = lengths[head];
    const float query = q[head * 32 + lane];
    float largest = -__int_as_float(0x7f800000), mass = 0, numerator = 0;
    for (int t = 0; t < length; ++t) {
        const int index = (head * capacity + t) * 32 + lane;
        float dot = query * keys[index];
        for (int step = 16; step; step /= 2) dot += __shfl_xor_sync(0xffffffffu, dot, step);
        const float score = dot * 0.1767766952966369f;
        const float next = fmaxf(largest, score);
        const float carry = expf(largest - next), weight = expf(score - next);
        numerator = numerator * carry + weight * values[index];
        mass = mass * carry + weight;
        largest = next;
    }
    out[head * 32 + lane] = length ? numerator / mass : 0.f;
}
