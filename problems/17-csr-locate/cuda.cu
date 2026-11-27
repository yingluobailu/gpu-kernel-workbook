extern "C" __global__ void solve(const int* offsets, const int* positions, int* owner, int* local,
                                 int rows, int queries) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= queries) return;
    const int p = positions[i];
    if (p < 0 || p >= offsets[rows]) {
        owner[i] = local[i] = -1;
        return;
    }
    int lo = 0, hi = rows;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (offsets[mid + 1] <= p) lo = mid + 1;
        else hi = mid;
    }
    owner[i] = lo;
    local[i] = p - offsets[lo];
}
