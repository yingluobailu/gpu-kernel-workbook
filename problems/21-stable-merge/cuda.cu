extern "C" __global__ void solve(const int* a, const int* b, int* values, int* origin, int m, int n) {
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= m + n) return;
    int lo = max(0, rank - n), hi = min(rank, m);
    while (lo < hi) {
        const int i = (lo + hi) / 2, j = rank - i;
        if (a[i] <= b[j - 1]) lo = i + 1;
        else hi = i;
    }
    const int i = lo, j = rank - i;
    const bool from_a = i < m && (j == n || a[i] <= b[j]);
    values[rank] = from_a ? a[i] : b[j];
    origin[rank] = from_a ? i : m + j;
}
