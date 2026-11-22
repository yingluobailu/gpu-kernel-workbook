extern "C" __global__ void solve(const int* src, int* counts, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned matches = __ballot_sync(0xffffffffu, i < n && src[i] > 0);
    if ((threadIdx.x & 31) == 0 && i < n) {
        counts[i / 32] = __popc(matches);
    }
}
