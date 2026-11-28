extern "C" __global__ void solve(const int* keys, int* order, int rows, int width) {
    __shared__ unsigned long long pairs[1024];
    const int base = blockIdx.x * width;
    int padded = 1;
    while (padded < width) padded *= 2;
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        unsigned long long item = ~0ull;
        if (i < width) {
            const unsigned biased = static_cast<unsigned>(keys[base + i]) ^ 0x80000000u;
            item = (static_cast<unsigned long long>(biased) << 32) | static_cast<unsigned>(i);
        }
        pairs[i] = item;
    }
    __syncthreads();
    for (int size = 2; size <= padded; size *= 2) {
        for (int gap = size / 2; gap; gap /= 2) {
            for (int i = threadIdx.x; i < padded; i += blockDim.x) {
                const int partner = i ^ gap;
                if (partner <= i) continue;
                const auto a = pairs[i], b = pairs[partner];
                const bool ascending = (i & size) == 0;
                if (ascending ? a > b : a < b) {
                    pairs[i] = b;
                    pairs[partner] = a;
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        order[base + i] = static_cast<unsigned>(pairs[i]);
}
