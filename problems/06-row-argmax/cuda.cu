struct Candidate {
    float value;
    int column;
};

__device__ Candidate better(Candidate a, Candidate b) {
    return b.value > a.value || (b.value == a.value && b.column < a.column) ? b : a;
}

__device__ Candidate warp_best(Candidate candidate) {
    for (int step = 16; step > 0; step /= 2) {
        Candidate other = {
            __shfl_down_sync(0xffffffffu, candidate.value, step),
            __shfl_down_sync(0xffffffffu, candidate.column, step)
        };
        candidate = better(candidate, other);
    }
    return candidate;
}

extern "C" __global__ void solve(const float* src, float* values, int* indices, int rows, int cols) {
    __shared__ Candidate shared[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const Candidate neutral = {-__int_as_float(0x7f800000), 0x7fffffff};
    Candidate candidate = neutral;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        candidate = better(candidate, {src[row * cols + i], i});
    }
    candidate = warp_best(candidate);
    if (lane == 0) shared[warp] = candidate;
    __syncthreads();
    if (warp == 0) {
        candidate = warp_best(lane < 8 ? shared[lane] : neutral);
        if (lane == 0) {
            values[row] = candidate.value;
            indices[row] = candidate.column;
        }
    }
}
