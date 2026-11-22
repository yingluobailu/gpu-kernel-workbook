extern "C" __global__ void solve(const int* src, int* dst, int n) {
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int original = i < n ? src[i] : 0;
    int inclusive = original;
    for (int offset = 1; offset < 32; offset *= 2) {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    if (i < n) {
        dst[i] = inclusive - original;
    }
}
