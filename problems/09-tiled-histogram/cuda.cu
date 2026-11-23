constexpr int bins = 64;
constexpr int tile = 1024;
constexpr int warps = 8;

extern "C" __global__ void solve(const int* labels, int* partials, int n, int tiles) {
    __shared__ int local[warps][bins];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    for (int i = tid; i < warps * bins; i += blockDim.x) {
        local[i / bins][i % bins] = 0;
    }
    __syncthreads();

    const int first = blockIdx.x * tile + tid;
    for (int offset = 0; offset < tile; offset += blockDim.x) {
        const int index = first + offset;
        if (index < n) {
            const int label = labels[index];
            if (label >= 0 && label < bins) atomicAdd(&local[warp][label], 1);
        }
    }
    __syncthreads();

    if (tid < bins) {
        int count = 0;
        for (int w = 0; w < warps; ++w) count += local[w][tid];
        partials[tid * tiles + blockIdx.x] = count;
    }
}

__device__ int warp_sum(int value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void merge(const int* partials, int* histogram, int tiles) {
    __shared__ int subtotals[warps];
    const int bin = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid / 32;
    int subtotal = 0;
    for (int i = tid; i < tiles; i += blockDim.x) subtotal += partials[bin * tiles + i];

    subtotal = warp_sum(subtotal);
    if (lane == 0) subtotals[warp] = subtotal;
    __syncthreads();
    if (warp == 0) {
        subtotal = warp_sum(lane < warps ? subtotals[lane] : 0);
        if (lane == 0) histogram[bin] = subtotal;
    }
}
