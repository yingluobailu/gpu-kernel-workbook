extern "C" __global__ void solve(const int* values, const int* heads, int* out, int rows, int width) {
    const int base = blockIdx.x * width, lane = threadIdx.x;
    __shared__ int sums[128], flags[128];
    int carry = 0;
    for (int tile = 0; tile < width; tile += 128) {
        const int c = tile + lane;
        int sum = c < width ? values[base + c] : 0;
        int head = c < width ? heads[base + c] : 0;
        for (int gap = 1; gap < 128; gap *= 2) {
            sums[lane] = sum;
            flags[lane] = head;
            __syncthreads();
            if (lane >= gap) {
                if (!head) sum += sums[lane - gap];
                head |= flags[lane - gap];
            }
            __syncthreads();
        }
        if (!head) sum += carry;
        if (c < width) out[base + c] = sum;
        sums[lane] = sum;
        __syncthreads();
        carry = sums[127];
        __syncthreads();
    }
}
