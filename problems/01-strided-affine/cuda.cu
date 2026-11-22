extern "C" __global__ void solve(const float* src, float* dst, int n, int stride, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = 2.0f * src[offset + i * stride] - 3.0f;
    }
}
