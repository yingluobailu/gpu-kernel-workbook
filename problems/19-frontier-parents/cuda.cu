extern "C" __global__ void solve(const int* offsets, const int* sources, const int* state,
                                 int* parent, int vertices, int degree) {
    const int dst = blockIdx.x, tid = threadIdx.x;
    if (state[dst] != 0) {
        if (tid == 0) parent[dst] = -1;
        return;
    }
    int best = vertices;
    for (int j = offsets[dst] + tid; j < offsets[dst + 1]; j += blockDim.x) {
        const int src = sources[j];
        if (state[src] == 1) best = min(best, src);
    }
    __shared__ int candidates[128];
    candidates[tid] = best;
    __syncthreads();
    for (int step = 64; step; step /= 2) {
        if (tid < step) candidates[tid] = min(candidates[tid], candidates[tid + step]);
        __syncthreads();
    }
    if (tid == 0) parent[dst] = candidates[0] == vertices ? -1 : candidates[0];
}
