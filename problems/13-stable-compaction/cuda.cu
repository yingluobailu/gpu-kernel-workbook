constexpr int tile_width = 256;
constexpr int warp_count = tile_width / 32;

__device__ int warp_inclusive(int item, int lane) {
    for (int step = 1; step < 32; step *= 2) {
        const int previous = __shfl_up_sync(0xffffffffu, item, step);
        if (lane >= step) item += previous;
    }
    return item;
}

__device__ int block_inclusive(int item, int* warp_totals, int& total) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    item = warp_inclusive(item, lane);
    if (lane == 31) warp_totals[warp] = item;
    __syncthreads();
    if (warp == 0) {
        const int subtotal = warp_inclusive(lane < warp_count ? warp_totals[lane] : 0, lane);
        if (lane < warp_count) warp_totals[lane] = subtotal;
    }
    __syncthreads();
    const int result = item + (warp ? warp_totals[warp - 1] : 0);
    total = warp_totals[warp_count - 1];
    __syncthreads();
    return result;
}

extern "C" __global__ void solve(const int* src, int* counts, int n) {
    __shared__ int warp_totals[warp_count];
    const int i = blockIdx.x * tile_width + threadIdx.x;
    const int keep = i < n && src[i] > 0;
    int total;
    block_inclusive(keep, warp_totals, total);
    if (threadIdx.x == 0) counts[blockIdx.x] = total;
}

extern "C" __global__ void scan(const int* counts, int* offsets, int* total, int tiles) {
    __shared__ int warp_totals[warp_count];
    int carry = 0;
    for (int base = 0; base < tiles; base += tile_width) {
        const int i = base + threadIdx.x;
        const int item = i < tiles ? counts[i] : 0;
        int subtotal;
        const int prefix = block_inclusive(item, warp_totals, subtotal);
        if (i < tiles) offsets[i] = carry + prefix - item;
        carry += subtotal;
    }
    if (threadIdx.x == 0) total[0] = carry;
}

extern "C" __global__ void scatter(const int* src, const int* offsets, const int* total,
                                   int* values, int* indices, int n) {
    __shared__ int warp_totals[warp_count];
    const int i = blockIdx.x * tile_width + threadIdx.x;
    const int value = i < n ? src[i] : 0;
    const int keep = value > 0;
    int subtotal;
    const int prefix = block_inclusive(keep, warp_totals, subtotal);
    if (keep) {
        const int destination = offsets[blockIdx.x] + prefix - 1;
        values[destination] = value;
        indices[destination] = i;
    }
    if (i < n && i >= total[0]) {
        values[i] = 0;
        indices[i] = -1;
    }
}
