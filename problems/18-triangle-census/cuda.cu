extern "C" __global__ void solve(const int* offsets, const int* adj, int* counts, int vertices, int percent) {
    const int u = blockIdx.x, tid = threadIdx.x;
    const int first = offsets[u], end = offsets[u + 1];
    int subtotal = 0;
    for (int j = first; j < end; ++j) {
        const int v = adj[j];
        for (int p = offsets[v] + tid; p < offsets[v + 1]; p += blockDim.x) {
            const int w = adj[p];
            int lo = j + 1, hi = end;
            while (lo < hi) {
                const int mid = (lo + hi) / 2;
                if (adj[mid] < w) lo = mid + 1;
                else hi = mid;
            }
            subtotal += lo < end && adj[lo] == w;
        }
    }
    __shared__ int partial[64];
    partial[tid] = subtotal;
    __syncthreads();
    for (int stride = 32; stride; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) counts[u] = partial[0];
}
