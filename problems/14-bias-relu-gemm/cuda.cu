extern "C" __global__ void solve(const float* a, const float* b, const float* bias,
                                 float* out, int m, int n, int k) {
    __shared__ float left[16][16], right[16][16];
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = blockIdx.y * 16 + y, col = blockIdx.x * 16 + x;
    float sum = 0;
    for (int base = 0; base < k; base += 16) {
        left[y][x] = row < m && base + x < k ? a[row * k + base + x] : 0;
        right[y][x] = base + y < k && col < n ? b[(base + y) * n + col] : 0;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 16; ++i) sum += left[y][i] * right[i][x];
        __syncthreads();
    }
    if (row < m && col < n) out[row * n + col] = fmaxf(0.f, sum + bias[col]);
}
